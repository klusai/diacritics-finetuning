"""BERT-based token classification for Romanian diacritic restoration.

Uses the Naplava et al. instruction-per-subword approach: for each subword
token, predict a diacritization instruction that maps the undiacritized form
to the diacritized one. Instructions are generated automatically from aligned
training pairs.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "readerbench/RoBERT-base"


def build_instruction_vocab(
    pairs: list[tuple[str, str]], tokenizer, min_count: int = 2
) -> dict[str, int]:
    """Build instruction vocabulary from (stripped, diacritized) pairs.

    An instruction is the transformation needed to convert a stripped
    subword token to its diacritized form. Most common instruction is
    identity (no change needed).
    """
    instruction_counts = Counter()

    for stripped, diacritized in pairs:
        stripped_tokens = tokenizer.tokenize(stripped)
        diac_tokens = tokenizer.tokenize(diacritized)

        if len(stripped_tokens) != len(diac_tokens):
            instruction_counts["<IDENTITY>"] += len(stripped_tokens)
            continue

        for st, dt in zip(stripped_tokens, diac_tokens):
            instr = _compute_instruction(st, dt)
            instruction_counts[instr] += 1

    vocab = {"<PAD>": 0, "<IDENTITY>": 1}
    for instr, count in instruction_counts.most_common():
        if instr in vocab:
            continue
        if count >= min_count:
            vocab[instr] = len(vocab)

    logger.info("Instruction vocab: %d instructions (from %d unique, min_count=%d)",
                len(vocab), len(instruction_counts), min_count)
    return vocab


def _compute_instruction(stripped_token: str, diac_token: str) -> str:
    """Compute the diacritization instruction for a token pair."""
    s = stripped_token.replace("##", "").replace("▁", "")
    d = diac_token.replace("##", "").replace("▁", "")

    if s == d:
        return "<IDENTITY>"

    changes = []
    min_len = min(len(s), len(d))
    for i in range(min_len):
        if s[i] != d[i]:
            changes.append(f"{i}:{s[i]}>{d[i]}")

    if not changes:
        return "<IDENTITY>"
    return "|".join(changes)


def apply_instruction(token: str, instruction: str) -> str:
    """Apply a diacritization instruction to a token."""
    if instruction == "<IDENTITY>" or instruction == "<PAD>":
        return token

    clean = token.replace("##", "").replace("▁", "")
    prefix = token[:len(token) - len(clean)]
    chars = list(clean)

    for change in instruction.split("|"):
        parts = change.split(":")
        if len(parts) != 2:
            continue
        pos = int(parts[0])
        mapping = parts[1].split(">")
        if len(mapping) != 2:
            continue
        if pos < len(chars):
            chars[pos] = mapping[1]

    return prefix + "".join(chars)


class InstructionDataset(Dataset):
    """Dataset for BERT token classification with diacritization instructions."""

    def __init__(self, pairs, tokenizer, instruction_vocab, max_length=192):
        self.encodings = []
        self.labels = []
        self.tokenizer = tokenizer
        self.instruction_vocab = instruction_vocab
        self.max_length = max_length

        identity_id = instruction_vocab.get("<IDENTITY>", 1)

        for stripped, diacritized in pairs:
            enc = tokenizer(
                stripped, truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt",
            )

            stripped_tokens = tokenizer.tokenize(stripped)[:max_length - 2]
            diac_tokens = tokenizer.tokenize(diacritized)[:max_length - 2]

            label_ids = [0]  # CLS
            if len(stripped_tokens) == len(diac_tokens):
                for st, dt in zip(stripped_tokens, diac_tokens):
                    instr = _compute_instruction(st, dt)
                    label_ids.append(instruction_vocab.get(instr, identity_id))
            else:
                label_ids.extend([identity_id] * len(stripped_tokens))

            while len(label_ids) < max_length - 1:
                label_ids.append(0)  # padding
            label_ids.append(0)  # SEP
            label_ids = label_ids[:max_length]

            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})
            self.labels.append(torch.tensor(label_ids, dtype=torch.long))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v for k, v in self.encodings[idx].items()}
        item["labels"] = self.labels[idx]
        return item


class BERTClassifierModel:
    """High-level wrapper for BERT-based diacritic restoration."""

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 192):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.instruction_vocab = None
        self.idx_to_instruction = None

    def train(self, pairs: list[tuple[str, str]], val_pairs: list[tuple[str, str]] | None = None,
              epochs: int = 5, lr: float = 3e-5, batch_size: int = 16):
        from transformers import (
            AutoModelForTokenClassification, AutoTokenizer,
            Trainer, TrainingArguments,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.instruction_vocab = build_instruction_vocab(pairs, self.tokenizer)
        self.idx_to_instruction = {v: k for k, v in self.instruction_vocab.items()}

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=len(self.instruction_vocab),
        )

        logger.info("Building training dataset (%d pairs)...", len(pairs))
        train_ds = InstructionDataset(pairs, self.tokenizer, self.instruction_vocab, self.max_length)

        eval_ds = None
        if val_pairs:
            logger.info("Building validation dataset (%d pairs)...", len(val_pairs))
            eval_ds = InstructionDataset(val_pairs[:5000], self.tokenizer, self.instruction_vocab, self.max_length)

        args = TrainingArguments(
            output_dir="artifacts/bert_training",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=100,
            eval_strategy="epoch" if eval_ds else "no",
            save_strategy="epoch",
            load_best_model_at_end=bool(eval_ds),
            fp16=False,
            dataloader_num_workers=0,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model, args=args,
            train_dataset=train_ds, eval_dataset=eval_ds,
        )

        logger.info("Starting BERT training: %s, %d epochs, lr=%s", self.model_name, epochs, lr)
        trainer.train()

    def predict(self, text: str) -> str:
        """Predict diacritized text from stripped input."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        self.model.eval()
        device = next(self.model.parameters()).device

        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0).cpu().tolist())

        restored_tokens = []
        for tok, pid in zip(tokens, pred_ids):
            if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"):
                continue
            instr = self.idx_to_instruction.get(pid, "<IDENTITY>")
            restored_tokens.append(apply_instruction(tok, instr))

        return self.tokenizer.convert_tokens_to_string(restored_tokens)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "model")
        with open(path / "instruction_vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.instruction_vocab, f, ensure_ascii=False)
        logger.info("Saved BERT model to %s", path)

    @classmethod
    def load(cls, path: Path, model_name: str = DEFAULT_MODEL) -> "BERTClassifierModel":
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        inst = cls(model_name)
        inst.model = AutoModelForTokenClassification.from_pretrained(path / "model")
        inst.tokenizer = AutoTokenizer.from_pretrained(path / "model")
        with open(path / "instruction_vocab.json", encoding="utf-8") as f:
            inst.instruction_vocab = json.load(f)
        inst.idx_to_instruction = {int(v): k for k, v in inst.instruction_vocab.items()}
        logger.info("Loaded BERT model from %s", path)
        return inst
