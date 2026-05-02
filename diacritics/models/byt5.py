"""ByT5 byte-level seq2seq for Romanian diacritic restoration.

Uses google/byt5-small (300M) with byte-level tokenization, which
avoids subword alignment issues and is expected to be more robust
to noisy input than subword-based models.
"""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/byt5-small"


class ByT5Dataset(Dataset):
    """Dataset for ByT5 seq2seq diacritic restoration."""

    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_length: int = 384):
        self.inputs = []
        self.targets = []

        for stripped, diacritized in pairs:
            input_enc = tokenizer(
                stripped, truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt",
            )
            target_enc = tokenizer(
                diacritized, truncation=True, max_length=max_length,
                padding="max_length", return_tensors="pt",
            )

            labels = target_enc["input_ids"].squeeze(0).clone()
            labels[labels == tokenizer.pad_token_id] = -100

            self.inputs.append({
                "input_ids": input_enc["input_ids"].squeeze(0),
                "attention_mask": input_enc["attention_mask"].squeeze(0),
                "labels": labels,
            })

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class ByT5Model:
    """High-level wrapper for ByT5-based diacritic restoration."""

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 384):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def train(self, pairs: list[tuple[str, str]], val_pairs: list[tuple[str, str]] | None = None,
              epochs: int = 5, lr: float = 1e-3, batch_size: int = 8, use_cpu: bool = True):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        logger.info("Building training dataset (%d pairs)...", len(pairs))
        train_ds = ByT5Dataset(pairs, self.tokenizer, self.max_length)

        eval_ds = None
        if val_pairs:
            logger.info("Building validation dataset (%d pairs)...", len(val_pairs[:5000]))
            eval_ds = ByT5Dataset(val_pairs[:5000], self.tokenizer, self.max_length)

        args = Seq2SeqTrainingArguments(
            output_dir="artifacts/byt5_training",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=100,
            eval_strategy="epoch" if eval_ds else "no",
            save_strategy="epoch",
            load_best_model_at_end=bool(eval_ds),
            predict_with_generate=True,
            generation_max_length=self.max_length,
            fp16=False,
            use_cpu=use_cpu,
            dataloader_num_workers=0,
            gradient_accumulation_steps=2,
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=self.model, args=args,
            train_dataset=train_ds, eval_dataset=eval_ds,
        )

        device_label = "CPU" if use_cpu else "MPS"
        logger.info("Starting ByT5 training on %s: %s, %d epochs, lr=%s",
                     device_label, self.model_name, epochs, lr)
        trainer.train()

    def predict(self, text: str) -> str:
        """Generate diacritized text from stripped input."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        self.model.eval()
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=1,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "model")
        logger.info("Saved ByT5 model to %s", path)

    @classmethod
    def load(cls, path: Path, model_name: str = DEFAULT_MODEL) -> "ByT5Model":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        inst = cls(model_name)
        inst.model = AutoModelForSeq2SeqLM.from_pretrained(path / "model")
        inst.tokenizer = AutoTokenizer.from_pretrained(path / "model")
        logger.info("Loaded ByT5 model from %s", path)
        return inst
