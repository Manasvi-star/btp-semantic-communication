"""
Universal Transformer (UT) semantic communication model with Adaptive
Computation Time (ACT), following:

    Q. Zhou et al., "Semantic Communication with Adaptive Universal
    Transformer", arXiv:2108.09119.

Additions beyond the paper (our extensions):
    * SNR-aware halting: the ACT halting unit can optionally receive the
      channel SNR as an extra input feature, so the number of recurrent
      steps can depend explicitly on channel quality (`snr_aware_halting`).
    * Per-token step statistics are returned so compute cost can be
      measured and compared with fixed-depth DeepSC.

Also contains the channel models (AWGN and Rayleigh block fading with
perfect CSI at the receiver) and a thin wrapper around the existing
`DeepSCBaseline` so both models share the same interface:

    logits, aux = model(input_ids, attention_mask, snr_db, channel)
    aux = {"ponder": scalar tensor, "enc_steps": [B, L], "dec_steps": [B, L]}
"""

import math

import torch
import torch.nn as nn

from deepsc_replication import (
    ChannelDecoder,
    ChannelEncoder,
    DeepSCBaseline,
    PositionalEncoding,
    power_normalize,
    snr_to_noise_std,
)


# --------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------
def awgn_channel(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    noise_std = snr_to_noise_std(snr_db)
    return x + torch.randn_like(x) * noise_std


def rayleigh_channel(x: torch.Tensor, snr_db: float, eps: float = 1e-6) -> torch.Tensor:
    """Rayleigh block fading, y = h x + n, with zero-forcing equalisation.

    The last dimension (channel_dim) is treated as pairs of real numbers,
    i.e. channel_dim / 2 complex symbols per token. One complex gain
    h ~ CN(0, 1) is drawn per sentence (constant over the sentence, as in
    the paper: "channel gain that remains constant throughout the
    transmission"). Perfect CSI is assumed at the receiver, matching the
    paper's assumption that CSI is available from feedback.
    """
    if x.size(-1) % 2 != 0:
        raise ValueError("channel_dim must be even for the Rayleigh channel (I/Q pairs).")
    noise_std = snr_to_noise_std(snr_db)
    batch = x.size(0)
    shape = [batch] + [1] * (x.dim() - 1)
    h_re = torch.randn(shape, device=x.device) / math.sqrt(2.0)
    h_im = torch.randn(shape, device=x.device) / math.sqrt(2.0)

    x_re, x_im = x[..., 0::2], x[..., 1::2]
    y_re = h_re * x_re - h_im * x_im + torch.randn_like(x_re) * noise_std
    y_im = h_im * x_re + h_re * x_im + torch.randn_like(x_im) * noise_std

    # Zero-forcing: multiply by conj(h) / |h|^2
    h_pow = h_re ** 2 + h_im ** 2 + eps
    xh_re = (y_re * h_re + y_im * h_im) / h_pow
    xh_im = (y_im * h_re - y_re * h_im) / h_pow

    out = torch.empty_like(x)
    out[..., 0::2] = xh_re
    out[..., 1::2] = xh_im
    return out


def apply_channel(x: torch.Tensor, snr_db: float, channel: str) -> torch.Tensor:
    if channel == "awgn":
        return awgn_channel(x, snr_db)
    if channel == "rayleigh":
        return rayleigh_channel(x, snr_db)
    raise ValueError(f"Unknown channel: {channel}")


# --------------------------------------------------------------------------
# Adaptive Computation Time (paper Algorithm 1)
# --------------------------------------------------------------------------
class ACT(nn.Module):
    """Per-position ACT halting, as used in the Universal Transformer.

    At every recurrent step a dense+sigmoid unit predicts a halting
    probability p for each token. A token stops once its cumulative halting
    probability exceeds `threshold` or `max_steps` is reached. The output is
    the halting-probability-weighted average of the states, and the ponder
    cost rho = N + R (number of updates + remainder) is returned.
    """

    def __init__(self, d_model: int, threshold: float, max_steps: int, snr_aware: bool):
        super().__init__()
        self.threshold = threshold
        self.max_steps = max_steps
        self.snr_aware = snr_aware
        in_dim = d_model + (1 if snr_aware else 0)
        self.halting_unit = nn.Linear(in_dim, 1)
        # UT/ACT practice: positive bias so early steps do not halt immediately
        nn.init.constant_(self.halting_unit.bias, 1.0)
        # Step ("depth") embedding, the UT "timing signal" over recurrent steps
        self.step_embedding = nn.Embedding(max_steps, d_model)

    def forward(self, state, step_fn, pad_mask, snr_feature=None):
        """
        state:       [B, L, D]
        step_fn:     callable(state) -> new state, the shared transformer block
        pad_mask:    [B, L] bool, True where padding
        snr_feature: [B] float, normalised SNR (only used when snr_aware)
        """
        bsz, seq_len, _ = state.shape
        device = state.device

        halting_prob = torch.zeros(bsz, seq_len, device=device)
        remainders = torch.zeros(bsz, seq_len, device=device)
        n_updates = torch.zeros(bsz, seq_len, device=device)
        if pad_mask is not None:
            # padded positions are treated as already halted
            halting_prob = halting_prob.masked_fill(pad_mask, 1.0)
        previous_state = torch.zeros_like(state)

        for step in range(self.max_steps):
            state = state + self.step_embedding.weight[step].view(1, 1, -1)

            halt_in = state
            if self.snr_aware:
                snr_col = snr_feature.view(bsz, 1, 1).expand(bsz, seq_len, 1)
                halt_in = torch.cat([state, snr_col], dim=-1)
            p = torch.sigmoid(self.halting_unit(halt_in)).squeeze(-1)

            running = (halting_prob < 1.0).float()
            if pad_mask is not None:
                running = running * (~pad_mask).float()

            if step == self.max_steps - 1:
                # force every still-running position to halt at the last step
                new_halted = running
                still_running = torch.zeros_like(running)
            else:
                crossed = (halting_prob + p * running) > self.threshold
                new_halted = crossed.float() * running
                still_running = (~crossed).float() * running

            halting_prob = halting_prob + p * still_running
            remainders = remainders + new_halted * (1.0 - halting_prob)
            halting_prob = halting_prob + new_halted * remainders
            n_updates = n_updates + still_running + new_halted

            update_weights = (p * still_running + new_halted * remainders).unsqueeze(-1)

            state = step_fn(state)
            previous_state = state * update_weights + previous_state * (1.0 - update_weights)

            if still_running.sum() == 0:
                break

        valid = (~pad_mask).float() if pad_mask is not None else torch.ones_like(n_updates)
        ponder = ((n_updates + remainders) * valid).sum() / valid.sum().clamp(min=1.0)
        return previous_state, ponder, n_updates


# --------------------------------------------------------------------------
# Universal Transformer semantic communication system
# --------------------------------------------------------------------------
class UTSemCom(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        channel_dim: int,
        channel_hidden: int,
        channel_decoder_hidden: int,
        max_len: int,
        pad_id: int,
        max_steps: int = 5,
        act_threshold: float = 0.9,
        snr_aware_halting: bool = False,
        dropout: float = 0.1,
        snr_norm_db: float = 20.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.max_steps = max_steps
        self.snr_aware_halting = snr_aware_halting
        self.snr_norm_db = snr_norm_db

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # One shared block each for encoder and decoder, applied recurrently
        self.enc_block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.dec_block = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.enc_act = ACT(d_model, act_threshold, max_steps, snr_aware_halting)
        self.dec_act = ACT(d_model, act_threshold, max_steps, snr_aware_halting)
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)

        self.channel_encoder = ChannelEncoder(d_model, channel_hidden, channel_dim)
        self.channel_decoder = ChannelDecoder(channel_dim, d_model, channel_decoder_hidden)
        self.output_layer = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def _snr_feature(self, snr_db: float, batch: int, device) -> torch.Tensor:
        return torch.full((batch,), snr_db / self.snr_norm_db, device=device)

    def encode_and_transmit(self, input_ids, attention_mask, snr_db: float, channel: str):
        src_pad = attention_mask == 0
        emb = self.pos_encoder(self.embedding(input_ids) * math.sqrt(self.d_model))
        snr_feat = self._snr_feature(snr_db, input_ids.size(0), input_ids.device)

        def enc_step(x):
            return self.enc_block(x, src_key_padding_mask=src_pad)

        semantic, enc_ponder, enc_steps = self.enc_act(emb, enc_step, src_pad, snr_feat)
        semantic = self.enc_norm(semantic)

        symbols = power_normalize(self.channel_encoder(semantic))
        received = apply_channel(symbols, snr_db, channel)
        memory = self.channel_decoder(received)
        return memory, src_pad, enc_ponder, enc_steps, snr_feat

    def decode(self, tgt_ids, memory, src_pad, snr_feat, tgt_pad=None):
        dec_emb = self.pos_encoder(self.embedding(tgt_ids) * math.sqrt(self.d_model))
        causal = self._causal_mask(tgt_ids.size(1), tgt_ids.device)

        def dec_step(x):
            return self.dec_block(
                x, memory, tgt_mask=causal,
                tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad,
            )

        out, dec_ponder, dec_steps = self.dec_act(dec_emb, dec_step, tgt_pad, snr_feat)
        return self.output_layer(self.dec_norm(out)), dec_ponder, dec_steps

    def forward(self, input_ids, attention_mask, snr_db: float, channel: str = "awgn"):
        memory, src_pad, enc_ponder, enc_steps, snr_feat = self.encode_and_transmit(
            input_ids, attention_mask, snr_db, channel
        )
        logits, dec_ponder, dec_steps = self.decode(input_ids, memory, src_pad, snr_feat, tgt_pad=src_pad)
        aux = {"ponder": enc_ponder + dec_ponder, "enc_steps": enc_steps, "dec_steps": dec_steps}
        return logits, aux

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, snr_db: float, sos_id: int, eos_id: int, channel: str = "awgn"):
        """Batched greedy decoding. Returns (token ids [B, T], step stats)."""
        self.eval()
        memory, src_pad, _, enc_steps, snr_feat = self.encode_and_transmit(
            input_ids, attention_mask, snr_db, channel
        )
        bsz = input_ids.size(0)
        generated = torch.full((bsz, 1), sos_id, dtype=torch.long, device=input_ids.device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
        dec_step_log = []

        for _ in range(self.max_len - 1):
            logits, _, dec_steps = self.decode(generated, memory, src_pad, snr_feat)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            # steps used for the newly generated position, only for unfinished rows
            dec_step_log.append(torch.where(finished, torch.nan, dec_steps[:, -1]))
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        enc_valid = (~src_pad).float()
        enc_mean = (enc_steps * enc_valid).sum(1) / enc_valid.sum(1).clamp(min=1.0)
        dec_mean = torch.nanmean(torch.stack(dec_step_log, dim=1), dim=1)
        return generated, {"enc_steps": enc_mean, "dec_steps": dec_mean}


# --------------------------------------------------------------------------
# DeepSC baseline with the same interface (and selectable channel)
# --------------------------------------------------------------------------
class DeepSCWrapper(DeepSCBaseline):
    """Existing DeepSC baseline, extended to support the Rayleigh channel and
    to return the same (logits, aux) interface as UTSemCom."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_enc_layers = len(self.semantic_encoder.layers)
        self.num_dec_layers = len(self.semantic_decoder.layers)

    def _transmit(self, input_ids, attention_mask, snr_db, channel):
        emb = self.pos_encoder(self.embedding(input_ids) * math.sqrt(self.d_model))
        src_pad = attention_mask == 0
        feats = self.semantic_encoder(emb, src_key_padding_mask=src_pad)
        symbols = power_normalize(self.channel_encoder(feats))
        memory = self.channel_decoder(apply_channel(symbols, snr_db, channel))
        return memory, src_pad

    def forward(self, input_ids, attention_mask, snr_db: float, channel: str = "awgn"):
        memory, src_pad = self._transmit(input_ids, attention_mask, snr_db, channel)
        dec_emb = self.pos_encoder(self.embedding(input_ids) * math.sqrt(self.d_model))
        causal = self._causal_mask(input_ids.size(1), input_ids.device)
        out = self.semantic_decoder(
            dec_emb, memory, tgt_mask=causal,
            tgt_key_padding_mask=src_pad, memory_key_padding_mask=src_pad,
        )
        bsz, seq_len = input_ids.shape
        aux = {
            "ponder": torch.zeros((), device=input_ids.device),
            "enc_steps": torch.full((bsz, seq_len), float(self.num_enc_layers), device=input_ids.device),
            "dec_steps": torch.full((bsz, seq_len), float(self.num_dec_layers), device=input_ids.device),
        }
        return self.output_layer(out), aux

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, snr_db: float, sos_id: int, eos_id: int, channel: str = "awgn"):
        self.eval()
        memory, src_pad = self._transmit(input_ids, attention_mask, snr_db, channel)
        bsz = input_ids.size(0)
        generated = torch.full((bsz, 1), sos_id, dtype=torch.long, device=input_ids.device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
        for _ in range(self.max_len - 1):
            dec_emb = self.pos_encoder(self.embedding(generated) * math.sqrt(self.d_model))
            causal = self._causal_mask(generated.size(1), generated.device)
            out = self.semantic_decoder(dec_emb, memory, tgt_mask=causal, memory_key_padding_mask=src_pad)
            next_token = self.output_layer(out[:, -1, :]).argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break
        stats = {
            "enc_steps": torch.full((bsz,), float(self.num_enc_layers), device=input_ids.device),
            "dec_steps": torch.full((bsz,), float(self.num_dec_layers), device=input_ids.device),
        }
        return generated, stats
