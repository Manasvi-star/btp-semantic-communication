import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import DatasetDict, load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader

# Shared defaults aligned for DeepSC-style comparison
MAX_LEN = 30
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 1
PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"

_word_re = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")

def tokenize(text: str) -> List[str]:
    return _word_re.findall(text.lower())


def load_and_prepare(
    max_len: int = MAX_LEN,
    max_vocab_size: int = MAX_VOCAB_SIZE,
    min_freq: int = MIN_FREQ,
    seed: int = 42,
    dataset_name: str = "europarl",
    europarl_min_words: int = 4,
    europarl_max_words: int = 30,
    europarl_local_dir: str | None = None,
    split_protocol: str = "90_10_train_test",
) -> Tuple[DatasetDict, List[str], Dict[str, int], Dict[int, str], int]:
    """
    Load a text corpus, clean, build vocab, and encode to fixed length.

        Supported dataset_name values:
            - "europarl" (default): NLTK Europarl English sentences
            - "wikitext": HuggingFace WikiText-103 raw (fallback/legacy)

        europarl_local_dir:
            Optional local directory containing Europarl English text files (.txt / .en).
            If provided and dataset_name="europarl", this local data is used directly.
    """

    dataset_name = dataset_name.lower().strip()

    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        source_train = ds["train"]
    elif dataset_name == "europarl":
        lines: List[str] = []

        # Preferred path: use local Europarl files directly if user provides a directory.
        if europarl_local_dir:
            local_dir = Path(europarl_local_dir)
            if not local_dir.exists() or not local_dir.is_dir():
                raise ValueError(f"europarl_local_dir does not exist or is not a directory: {local_dir}")

            text_files = sorted(local_dir.glob("*.txt")) + sorted(local_dir.glob("*.en"))
            if not text_files:
                raise ValueError(
                    f"No .txt or .en files found in europarl_local_dir: {local_dir}"
                )

            for path in text_files:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        text = line.strip()
                        if text:
                            lines.append(text)
        else:
            try:
                from nltk.corpus import europarl_raw
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import nltk Europarl corpus. Install nltk and download europarl_raw, "
                    "or pass europarl_local_dir."
                ) from exc

            try:
                fileids = europarl_raw.fileids()
            except LookupError as exc:
                raise RuntimeError(
                    "NLTK europarl_raw corpus is not downloaded. Run: "
                    "import nltk; nltk.download('europarl_raw'), "
                    "or pass europarl_local_dir."
                ) from exc

            english_fileids = [
                fid for fid in fileids if ("english" in fid.lower() or fid.lower().endswith(".en"))
            ]
            use_fileids = english_fileids if english_fileids else fileids

            for fid in use_fileids:
                for sent in europarl_raw.sents(fid):
                    text = " ".join(sent).strip()
                    if text:
                        lines.append(text)

        source_train = HFDataset.from_dict({"text": lines})
    else:
        raise ValueError("dataset_name must be one of: 'wikitext', 'europarl'")

    # Filter out lines that are too short/long or contain unwanted chars
    def is_valid(sample):
        text = sample["text"].strip()   # Remove leading/trailing whitespace
        
        # Skip empty lines, section headers, and lines with special chars
        if re.match(r"^=+.*=+$", text):
            return False
        # Skip lines with special characters that are not suitable for tokenization
        if re.search(r"[@\{\}\[\]<>\\|]", text):
            return False
        
        # Check word count after tokenization to ensure it's within the desired range
        words = text.split()
        if dataset_name == "europarl":
            return europarl_min_words <= len(words) <= europarl_max_words
        return 4 <= len(words) <= 30

    # Apply the filter to the training split
    filtered = source_train.filter(is_valid)

    # Build vocab from the filtered training data 
    counts = Counter()
    for sample in filtered:
        counts.update(tokenize(sample["text"]))

    # Keep only the most common tokens that meet the frequency threshold and fit within the max vocab size
    most_common = [w for w, c in counts.most_common() if c >= min_freq][: max_vocab_size - 4]
    vocab = [PAD, SOS, EOS, UNK] + most_common
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for w, i in stoi.items()}
    pad_id = stoi[PAD]

    def encode(text: str) -> List[int]:
        tokens = tokenize(text)
        # Convert tokens to IDs, using UNK for out-of-vocab tokens, and ensure we have space for SOS and EOS
        ids = [stoi.get(tok, stoi[UNK]) for tok in tokens[: max_len - 1]]
        # Add SOS and EOS tokens, and pad/truncate to max_len
        ids = [stoi[SOS]] + ids + [stoi[EOS]]
        if len(ids) < max_len:
            ids += [pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len] # Truncate if it exceeds max_len (after adding SOS and EOS)
        return ids

    # DeepSC-compatible default: 90/10 train/test.
    # Keep a validation alias so existing notebooks that reference loaders['validation'] do not break.
    if split_protocol == "90_10_train_test":
        splits = filtered.train_test_split(test_size=0.1, seed=seed)
        dataset_splits = DatasetDict(
            {
                "train": splits["train"],
                "validation": splits["test"],
                "test": splits["test"],
            }
        )
    elif split_protocol == "80_10_10":
        splits = filtered.train_test_split(test_size=0.2, seed=seed)
        val_test = splits["test"].train_test_split(test_size=0.5, seed=seed)
        dataset_splits = DatasetDict(
            {
                "train": splits["train"],
                "validation": val_test["train"],
                "test": val_test["test"],
            }
        )
    else:
        raise ValueError("split_protocol must be one of: '90_10_train_test', '80_10_10'")

    def map_encode(batch):
        return {"input_ids": [encode(x) for x in batch["text"]]}

    # Encode the splits and remove the original text column to save memory , now the dataset is pure numeric and ready for training
    dataset_encoded = dataset_splits.map(map_encode, batched=True, remove_columns=["text"])
    return dataset_encoded, vocab, stoi, itos, pad_id


def _make_collate_fn(pad_id: int):
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        attention_mask = (input_ids != pad_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return collate_fn

# Get PyTorch dataloaders for each split with appropriate batching and shuffling
# The collate_fn ensures that we get tensors of input_ids and attention masks, ready for model input
# Note: shuffle only the training data, validation and test should be deterministic
def get_dataloaders(dataset_encoded: DatasetDict, pad_id: int, batch_size: int = 64, num_workers: int = 0):
    collate_fn = _make_collate_fn(pad_id)
    loaders = {}
    for split in ["train", "validation", "test"]:
        loaders[split] = DataLoader(
            dataset_encoded[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    return loaders


# Exported symbols for external use
__all__ = [
    "MAX_LEN",
    "MAX_VOCAB_SIZE",
    "MIN_FREQ",
    "PAD",
    "SOS",
    "EOS",
    "UNK",
    "tokenize",
    "load_and_prepare",
    "get_dataloaders",
]
