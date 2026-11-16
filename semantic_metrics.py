"""
Semantic similarity metric (SBERT), following seq2seq-SC
(Lee et al., arXiv:2210.15237):

    match(s_hat, s) = M(s) . M(s_hat)^T / (||M(s)|| ||M(s_hat)||)

where M is a Sentence-BERT model fine-tuned for semantic textual similarity.
BLEU measures word overlap; SBERT measures whether the *meaning* survived.

If `sentence-transformers` is not installed or the model cannot be
downloaded, `SBERTScorer.available` is False and scores are returned as NaN,
so the rest of the pipeline still runs.
"""

import math

DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SBERTScorer:
    def __init__(self, model_name: str = DEFAULT_SBERT_MODEL, device: str | None = None, batch_size: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size
        self.available = False
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=device)
            self.available = True
        except Exception as exc:  # missing package, no internet, etc.
            print(f"[SBERT] disabled ({type(exc).__name__}: {exc}). SBERT scores will be NaN.")

    def pairwise(self, refs: list[str], hyps: list[str]) -> list[float]:
        """Cosine similarity for each (reference, hypothesis) pair."""
        if not self.available:
            return [math.nan] * len(refs)
        import torch.nn.functional as F

        ref_emb = self._model.encode(refs, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False)
        hyp_emb = self._model.encode(hyps, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False)
        sims = F.cosine_similarity(ref_emb, hyp_emb, dim=-1)
        return [float(v) for v in sims.cpu()]
