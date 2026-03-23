import json
import pickle
import re
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def unicode_to_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    text = unicode_to_ascii(text.lower())
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9!?,;']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def length_filter(sentence: str, min_len: int, max_len: int) -> bool:
    n_tokens = len(sentence.split())
    return min_len < n_tokens < max_len


def tokenize(sentence: str, start_token: str, end_token: str, punct_to_keep: list[str], punct_to_remove: list[str]) -> list[str]:
    for punct in punct_to_keep:
        sentence = sentence.replace(punct, f" {punct}")
    for punct in punct_to_remove:
        sentence = sentence.replace(punct, "")
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return [start_token, *sentence.split(), end_token]


def build_vocab(tokenized_sentences: list[list[str]], special_tokens: list[str], min_count: int) -> dict[str, int]:
    counter = Counter(token for sent in tokenized_sentences for token in sent)
    token_to_idx = {tok: idx for idx, tok in enumerate(special_tokens)}

    for token, count in counter.items():
        if count >= min_count and token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode_sentences(tokenized_sentences: list[list[str]], token_to_idx: dict[str, int], unk_token: str) -> list[list[int]]:
    unk_id = token_to_idx[unk_token]
    return [[token_to_idx.get(tok, unk_id) for tok in sent] for sent in tokenized_sentences]


def _load_text_lines(input_dir: Path) -> list[str]:
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    lines = []
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines.extend(handle.readlines())
    return lines


def preprocess_and_save_europarl(
    input_dir: Path,
    train_pkl_path: Path,
    test_pkl_path: Path,
    vocab_json_path: Path,
    special_tokens: list[str],
    min_sentence_len: int,
    max_len: int,
    min_freq: int,
    punct_to_keep: list[str],
    punct_to_remove: list[str],
) -> dict:
    pad_token, start_token, end_token, unk_token = special_tokens

    raw_lines = _load_text_lines(input_dir)
    cleaned = [normalize_text(line) for line in raw_lines]
    cleaned = [line for line in cleaned if line and length_filter(line, min_sentence_len, max_len)]

    tokenized = [
        tokenize(
            sentence=line,
            start_token=start_token,
            end_token=end_token,
            punct_to_keep=punct_to_keep,
            punct_to_remove=punct_to_remove,
        )
        for line in cleaned
    ]

    token_to_idx = build_vocab(tokenized, special_tokens=special_tokens, min_count=min_freq)
    encoded = encode_sentences(tokenized, token_to_idx=token_to_idx, unk_token=unk_token)

    split_idx = int(0.9 * len(encoded))
    train_data = encoded[:split_idx]
    test_data = encoded[split_idx:]

    train_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    test_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_json_path.parent.mkdir(parents=True, exist_ok=True)

    with train_pkl_path.open("wb") as f:
        pickle.dump(train_data, f)
    with test_pkl_path.open("wb") as f:
        pickle.dump(test_data, f)

    vocab_payload = {
        "token_to_idx": token_to_idx,
        "special_tokens": {
            "pad": pad_token,
            "start": start_token,
            "end": end_token,
            "unk": unk_token,
        },
    }
    with vocab_json_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, indent=2)

    return {
        "raw_lines": len(raw_lines),
        "filtered_sentences": len(tokenized),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "vocab_size": len(token_to_idx),
    }


def load_preprocessed(
    train_pkl_path: Path,
    test_pkl_path: Path,
    vocab_json_path: Path,
) -> tuple[list[list[int]], list[list[int]], dict[str, int], dict[int, str]]:
    with train_pkl_path.open("rb") as f:
        train_data = pickle.load(f)
    with test_pkl_path.open("rb") as f:
        test_data = pickle.load(f)
    with vocab_json_path.open("r", encoding="utf-8") as f:
        vocab_payload = json.load(f)

    token_to_idx = vocab_payload["token_to_idx"]
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return train_data, test_data, token_to_idx, idx_to_token


class SeqDataset(Dataset):
    def __init__(self, encoded_samples: list[list[int]]):
        self.encoded_samples = encoded_samples

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx: int):
        sample = self.encoded_samples[idx]
        return torch.tensor(sample, dtype=torch.long), len(sample)


def collate_batch(batch, pad_id: int):
    sorted_batch = sorted(batch, key=lambda item: item[1], reverse=True)
    lengths = [item[1] for item in sorted_batch]
    max_len = lengths[0]

    batch_size = len(sorted_batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for row, (sequence, seq_len) in enumerate(sorted_batch):
        input_ids[row, :seq_len] = sequence
        attention_mask[row, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def build_dataloaders(
    train_data: list[list[int]],
    test_data: list[list[int]],
    pad_id: int,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    train_dataset = SeqDataset(train_data)
    test_dataset = SeqDataset(test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, pad_id=pad_id),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, pad_id=pad_id),
    )

    return {
        "train": train_loader,
        "validation": test_loader,
        "test": test_loader,
    }
