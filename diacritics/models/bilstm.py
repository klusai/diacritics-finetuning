"""Character-level BiLSTM for Romanian diacritic restoration.

Frames ADR as character-to-character transduction: for each input character,
predict the corresponding output character (which may be the same or a
diacritized variant).
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def build_char_vocab(texts: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build character vocabulary from a list of texts."""
    chars = set()
    for text in texts:
        chars.update(text)
    chars = sorted(chars)
    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for c in chars:
        char2idx[c] = len(char2idx)
    idx2char = {v: k for k, v in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """Pre-tokenized character dataset for efficient batching."""

    def __init__(self, pairs: list[tuple[str, str]], input_vocab: dict, output_vocab: dict,
                 max_len: int = 512):
        self.max_len = max_len
        logger.info("Pre-tokenizing %d pairs...", len(pairs))
        self.src_seqs = []
        self.tgt_seqs = []
        for src, tgt in pairs:
            min_len = min(len(src), len(tgt), max_len)
            src_ids = [input_vocab.get(c, 1) for c in src[:min_len]]
            tgt_ids = [output_vocab.get(c, 1) for c in tgt[:min_len]]
            self.src_seqs.append(torch.tensor(src_ids, dtype=torch.long))
            self.tgt_seqs.append(torch.tensor(tgt_ids, dtype=torch.long))
        logger.info("Pre-tokenization complete")

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, idx):
        return self.src_seqs[idx], self.tgt_seqs[idx]


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_padded = pad_sequence(srcs, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    return src_padded, tgt_padded, lengths


class BiLSTMRestorer(nn.Module):
    def __init__(self, input_vocab_size: int, output_vocab_size: int,
                 embedding_dim: int = 64, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_vocab_size)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


class BiLSTMModel:
    """High-level wrapper for training and inference."""

    def __init__(self, config=None):
        from diacritics.training.config import BiLSTMConfig
        self.config = config or BiLSTMConfig()
        self.model = None
        self.input_vocab = None
        self.output_vocab = None
        self.idx2char = None
        self.device = None

    def _select_device(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

    def train(self, pairs: list[tuple[str, str]], val_pairs: list[tuple[str, str]] | None = None):
        self._select_device()

        all_inputs = [p[0] for p in pairs]
        all_outputs = [p[1] for p in pairs]
        if val_pairs:
            all_inputs += [p[0] for p in val_pairs]
            all_outputs += [p[1] for p in val_pairs]

        self.input_vocab, _ = build_char_vocab(all_inputs)
        self.output_vocab, self.idx2char = build_char_vocab(all_outputs)

        logger.info("Input vocab: %d chars, Output vocab: %d chars",
                     len(self.input_vocab), len(self.output_vocab))

        train_ds = CharDataset(pairs, self.input_vocab, self.output_vocab,
                               max_len=self.config.max_seq_length)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size,
                                  shuffle=True, collate_fn=collate_fn,
                                  num_workers=4, persistent_workers=True)

        self.model = BiLSTMRestorer(
            input_vocab_size=len(self.input_vocab),
            output_vocab_size=len(self.output_vocab),
            embedding_dim=self.config.char_embedding_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info("BiLSTM parameters: %d (%.2f MB)", param_count, param_count * 4 / 1e6)

        total_batches = len(train_loader)
        for epoch in range(self.config.max_epochs):
            self.model.train()
            total_loss = 0
            n_batches = 0

            for batch_idx, (src, tgt, lengths) in enumerate(train_loader):
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                logits = self.model(src)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                if (batch_idx + 1) % 100 == 0:
                    logger.info("Epoch %d/%d [%d/%d]: running_loss=%.4f",
                                epoch + 1, self.config.max_epochs,
                                batch_idx + 1, total_batches,
                                total_loss / n_batches)

            avg_loss = total_loss / n_batches
            logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, self.config.max_epochs, avg_loss)

    def predict(self, text: str) -> str:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        self.model.eval()
        src_ids = torch.tensor(
            [[self.input_vocab.get(c, 1) for c in text]], dtype=torch.long
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(src_ids)
            pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        result = []
        for i, pid in enumerate(pred_ids[:len(text)]):
            result.append(self.idx2char.get(pid, text[i]))
        return "".join(result)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({
                "input_vocab": self.input_vocab,
                "output_vocab": self.output_vocab,
                "config": {
                    "embedding_dim": self.config.char_embedding_dim,
                    "hidden_size": self.config.hidden_size,
                    "num_layers": self.config.num_layers,
                    "dropout": self.config.dropout,
                },
            }, f, ensure_ascii=False)
        logger.info("Saved BiLSTM model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "BiLSTMModel":
        inst = cls()
        with open(path / "vocab.json", encoding="utf-8") as f:
            data = json.load(f)

        inst.input_vocab = data["input_vocab"]
        inst.output_vocab = data["output_vocab"]
        inst.idx2char = {int(v): k for k, v in inst.output_vocab.items()}
        cfg = data["config"]

        inst._select_device()
        inst.model = BiLSTMRestorer(
            input_vocab_size=len(inst.input_vocab),
            output_vocab_size=len(inst.output_vocab),
            embedding_dim=cfg["embedding_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(inst.device)
        inst.model.load_state_dict(torch.load(path / "model.pt", map_location=inst.device))
        inst.model.eval()
        logger.info("Loaded BiLSTM model from %s", path)
        return inst
