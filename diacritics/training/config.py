"""Training configuration dataclasses for all model families."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaseTrainingConfig:
    output_dir: str = "artifacts"
    seed: int = 42
    data_dir: str = "data/splits"

    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"

    max_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    fp16: bool = False  # default fp32 for MPS safety

    @property
    def train_path(self) -> Path:
        return Path(self.data_dir) / self.train_file

    @property
    def val_path(self) -> Path:
        return Path(self.data_dir) / self.val_file


@dataclass
class BiLSTMConfig(BaseTrainingConfig):
    char_embedding_dim: int = 64
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    use_crf: bool = False
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 10


@dataclass
class BERTConfig(BaseTrainingConfig):
    model_name: str = "dumitrescustefan/bert-base-romanian-cased-v1"
    learning_rate: float = 3e-5
    batch_size: int = 16
    max_epochs: int = 5
    max_seq_length: int = 256


@dataclass
class ByT5Config(BaseTrainingConfig):
    model_name: str = "google/byt5-small"
    learning_rate: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 5
    max_seq_length: int = 512
    use_cpu: bool = True  # prefer CPU over MPS for ByT5


@dataclass
class LoRAConfig(BaseTrainingConfig):
    model_name: str = "google/gemma-2-2b-it"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_epochs: int = 3
    max_seq_length: int = 512
    use_mlx: bool = True  # MLX for training, HF for eval

    @property
    def lora_config_dict(self) -> dict:
        return {
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
        }
