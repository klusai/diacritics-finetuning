"""Decoder-only LLM wrapper for Romanian diacritic restoration.

Supports both MLX (for LoRA training and inference) and standard
inference via any OpenAI-compatible API. The task is framed as
simple text completion: given stripped text, generate diacritized text.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

RESTORE_TEMPLATE = "Restore diacritics: {input}\n"


def format_completion_pair(stripped: str, diacritized: str) -> dict:
    """Format a training pair for mlx_lm completions format."""
    return {
        "prompt": RESTORE_TEMPLATE.format(input=stripped),
        "completion": diacritized + "\n",
    }


def prepare_mlx_dataset(
    pairs: list[tuple[str, str]],
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Prepare JSONL files in mlx_lm completions format.

    Creates train.jsonl and valid.jsonl in the output directory.
    """
    import random
    rng = random.Random(seed)

    formatted = [format_completion_pair(s, d) for s, d in pairs]
    rng.shuffle(formatted)

    val_size = int(len(formatted) * val_ratio)
    val_data = formatted[:val_size]
    train_data = formatted[val_size:]

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train.jsonl", train_data), ("valid.jsonl", val_data)]:
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Wrote %d items to %s", len(data), path)


class DecoderLMPredictor:
    """Inference wrapper for decoder-only models via OpenAI-compatible API.

    Works with Ollama, vLLM, or any server exposing /v1/chat/completions.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1",
                 temperature: float = 0.0):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="local")
        return self._client

    def predict(self, text: str) -> str:
        """Restore diacritics via the model API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": RESTORE_TEMPLATE.format(input=text),
                }],
                temperature=self.temperature,
                max_tokens=len(text) * 3,
            )
            result = response.choices[0].message.content.strip()
            result = result.split("\n")[0].strip()
            return result
        except Exception as e:
            logger.error("Prediction error for model %s: %s", self.model, e)
            return text

    def predict_batch(self, texts: list[str]) -> list[str]:
        """Predict diacritics for a batch of texts (sequential)."""
        return [self.predict(t) for t in texts]


class MLXDecoderLM:
    """MLX-based decoder LM for LoRA training and local inference.

    Uses mlx_lm for both training (via CLI) and inference.
    """

    def __init__(self, model_name: str, adapter_path: str | None = None):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def load(self):
        """Load the model (with optional LoRA adapter) for inference."""
        from mlx_lm import load
        if self.adapter_path:
            self._model, self._tokenizer = load(
                self.model_name, adapter_path=self.adapter_path
            )
        else:
            self._model, self._tokenizer = load(self.model_name)
        logger.info("Loaded %s (adapter: %s)", self.model_name, self.adapter_path)

    def predict(self, text: str, max_tokens: int = 512) -> str:
        """Generate diacritized text using MLX inference."""
        if self._model is None:
            self.load()

        from mlx_lm import generate
        prompt = RESTORE_TEMPLATE.format(input=text)
        result = generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=max_tokens,
        )
        result = result.split("\n")[0].strip()
        return result

    @staticmethod
    def train_lora(
        model_name: str,
        data_dir: str,
        output_dir: str,
        rank: int = 16,
        alpha: int = 32,
        epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        steps_per_eval: int = 200,
    ):
        """Launch LoRA training via mlx_lm CLI.

        This calls mlx_lm.lora as a subprocess for clean isolation.
        """
        import subprocess

        cmd = [
            "python", "-m", "mlx_lm.lora",
            "--model", model_name,
            "--data", data_dir,
            "--train",
            "--adapter-path", output_dir,
            "--lora-rank", str(rank),
            "--lora-alpha", str(alpha),
            "--num-epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--batch-size", str(batch_size),
            "--steps-per-eval", str(steps_per_eval),
            "--mask-prompt",
            "--grad-checkpoint",
        ]

        logger.info("Launching LoRA training: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"LoRA training failed with code {result.returncode}")
        logger.info("LoRA training complete. Adapters at %s", output_dir)

    @staticmethod
    def fuse_adapter(model_name: str, adapter_path: str, output_path: str):
        """Fuse LoRA adapter into the base model."""
        import subprocess

        cmd = [
            "python", "-m", "mlx_lm.fuse",
            "--model", model_name,
            "--adapter-path", adapter_path,
            "--save-path", output_path,
        ]

        logger.info("Fusing adapter: %s + %s -> %s", model_name, adapter_path, output_path)
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"Fuse failed with code {result.returncode}")
        logger.info("Fused model saved to %s", output_path)
