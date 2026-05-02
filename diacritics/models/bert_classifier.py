"""BERT-based character classification for Romanian diacritic restoration.

Tokenizes the diacritized (gold) text with BERT, then for each character
position, predicts which diacritic action to apply. This avoids the
subword alignment problem where stripped text tokenizes differently
from diacritized text in Romanian BERT.

The approach: feed the gold tokenization to BERT during training (teacher
forcing), and at inference time feed the stripped text. The model learns
to predict diacritics from context regardless of whether the input has them.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "readerbench/RoBERT-base"

DIACRITIC_ACTIONS = {
    "a": ["a", "ă", "â"],
    "i": ["i", "î"],
    "s": ["s", "ș"],
    "t": ["t", "ț"],
    "A": ["A", "Ă", "Â"],
    "I": ["I", "Î"],
    "S": ["S", "Ș"],
    "T": ["T", "Ț"],
}

ALL_LABELS = ["<PAD>", "<IDENTITY>"]
for base_char, variants in DIACRITIC_ACTIONS.items():
    for v in variants:
        if v != base_char:
            label = f"{base_char}>{v}"
            if label not in ALL_LABELS:
                ALL_LABELS.append(label)

LABEL2ID = {l: i for i, l in enumerate(ALL_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def char_label(stripped_char: str, gold_char: str) -> str:
    if stripped_char == gold_char:
        return "<IDENTITY>"
    key = f"{stripped_char}>{gold_char}"
    if key in LABEL2ID:
        return key
    return "<IDENTITY>"


def apply_label(char: str, label: str) -> str:
    if label == "<IDENTITY>" or label == "<PAD>":
        return char
    parts = label.split(">")
    if len(parts) == 2 and parts[0].lower() == char.lower():
        if char.isupper() and parts[1].islower():
            return parts[1].upper()
        return parts[1]
    return char


class CharClassificationDataset(Dataset):
    """Tokenize text with BERT, then produce per-character diacritic labels."""

    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_length: int = 256):
        self.items = []
        self.max_length = max_length

        for stripped, gold in pairs:
            enc = tokenizer(
                stripped, truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt",
                return_offsets_mapping=True,
            )

            offsets = enc.pop("offset_mapping").squeeze(0).tolist()
            enc.pop("token_type_ids", None)
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            labels = torch.zeros(max_length, dtype=torch.long)

            min_len = min(len(stripped), len(gold))
            for tok_idx, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                for char_pos in range(start, min(end, min_len)):
                    sc = stripped[char_pos] if char_pos < len(stripped) else ""
                    gc = gold[char_pos] if char_pos < len(gold) else sc
                    lbl = char_label(sc, gc)
                    if lbl != "<IDENTITY>":
                        labels[tok_idx] = LABEL2ID[lbl]
                        break  # one diacritic action per token

            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class BERTDiacriticHead(nn.Module):
    def __init__(self, bert_model, num_labels: int):
        super().__init__()
        self.bert = bert_model
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}


class BERTClassifierModel:
    """High-level wrapper for BERT-based diacritic restoration."""

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = None

    def _select_device(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

    def train(self, pairs: list[tuple[str, str]], val_pairs: list[tuple[str, str]] | None = None,
              epochs: int = 5, lr: float = 3e-5, batch_size: int = 16):
        self._select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        bert = AutoModel.from_pretrained(self.model_name)
        self.model = BERTDiacriticHead(bert, num_labels=len(ALL_LABELS)).to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info("BERT + head: %d params (%.1fM)", param_count, param_count / 1e6)

        logger.info("Building training dataset (%d pairs)...", len(pairs))
        train_ds = CharClassificationDataset(pairs, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        logger.info("Training: %d epochs, %d steps/epoch, %d total", epochs, len(train_loader), total_steps)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                if (batch_idx + 1) % 200 == 0:
                    logger.info("Epoch %d/%d [%d/%d]: loss=%.4f",
                                epoch + 1, epochs, batch_idx + 1, len(train_loader),
                                total_loss / n_batches)

            logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, epochs, total_loss / n_batches)

    def predict(self, text: str) -> str:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        self.model.eval()
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping").squeeze(0).tolist()
        enc.pop("token_type_ids", None)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc)["logits"]
            pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        chars = list(text)
        for tok_idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            label = ID2LABEL.get(pred_ids[tok_idx], "<IDENTITY>")
            if label not in ("<PAD>", "<IDENTITY>"):
                base_char = label.split(">")[0] if ">" in label else ""
                for char_pos in range(start, min(end, len(chars))):
                    if chars[char_pos].lower() == base_char.lower():
                        chars[char_pos] = apply_label(chars[char_pos], label)
                        break

        return "".join(chars)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        self.tokenizer.save_pretrained(path / "tokenizer")
        with open(path / "config.json", "w") as f:
            json.dump({"model_name": self.model_name, "max_length": self.max_length,
                        "num_labels": len(ALL_LABELS), "labels": ALL_LABELS}, f)
        logger.info("Saved BERT model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "BERTClassifierModel":
        with open(path / "config.json") as f:
            cfg = json.load(f)
        inst = cls(cfg["model_name"], cfg["max_length"])
        inst._select_device()
        inst.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        bert = AutoModel.from_pretrained(cfg["model_name"])
        inst.model = BERTDiacriticHead(bert, num_labels=cfg["num_labels"]).to(inst.device)
        inst.model.load_state_dict(torch.load(path / "model.pt", map_location=inst.device))
        inst.model.eval()
        logger.info("Loaded BERT model from %s", path)
        return inst
