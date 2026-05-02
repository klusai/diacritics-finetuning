"""LoRA training orchestration for decoder-only LMs on MLX.

Handles data preparation, training launch, adapter fusion, and evaluation
for the complete LoRA fine-tuning workflow.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from diacritics.models.decoder_lm import prepare_mlx_dataset, MLXDecoderLM

logger = logging.getLogger(__name__)


@dataclass
class LoRAExperiment:
    """Configuration for a single LoRA training experiment."""
    model_name: str
    experiment_name: str
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    steps_per_eval: int = 200


def run_lora_experiment(
    config: LoRAExperiment,
    data_pairs: list[tuple[str, str]],
    output_base: Path,
    skip_training: bool = False,
) -> Path:
    """Run a complete LoRA experiment: prepare data, train, fuse.

    Returns the path to the fused model directory.
    """
    exp_dir = output_base / config.experiment_name
    data_dir = exp_dir / "data"
    adapter_dir = exp_dir / "adapters"
    fused_dir = exp_dir / "fused"

    with open(exp_dir / "config.json", "w") as f:
        json.dump({
            "model_name": config.model_name,
            "experiment_name": config.experiment_name,
            "rank": config.rank,
            "alpha": config.alpha,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "n_training_pairs": len(data_pairs),
        }, f, indent=2)

    logger.info("Preparing data for %s (%d pairs)", config.experiment_name, len(data_pairs))
    prepare_mlx_dataset(data_pairs, data_dir)

    if not skip_training:
        logger.info("Starting LoRA training: %s", config.experiment_name)
        start = time.time()
        MLXDecoderLM.train_lora(
            model_name=config.model_name,
            data_dir=str(data_dir),
            output_dir=str(adapter_dir),
            rank=config.rank,
            alpha=config.alpha,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            steps_per_eval=config.steps_per_eval,
        )
        elapsed = time.time() - start
        logger.info("Training completed in %.1f min", elapsed / 60)

        with open(exp_dir / "training_time.json", "w") as f:
            json.dump({"elapsed_seconds": elapsed, "elapsed_minutes": elapsed / 60}, f)

    if adapter_dir.exists():
        logger.info("Fusing adapter for %s", config.experiment_name)
        MLXDecoderLM.fuse_adapter(config.model_name, str(adapter_dir), str(fused_dir))

    return fused_dir


STUDY_MODELS = [
    LoRAExperiment(
        model_name="Qwen/Qwen3-1.7B",
        experiment_name="qwen3-1.7b",
        rank=16, alpha=32, epochs=3, batch_size=4,
    ),
    LoRAExperiment(
        model_name="faur-ai/LLMic_v2",
        experiment_name="llmic-v2-3b",
        rank=16, alpha=32, epochs=3, batch_size=2,
    ),
    LoRAExperiment(
        model_name="Qwen/Qwen3-4B",
        experiment_name="qwen3-4b",
        rank=16, alpha=32, epochs=3, batch_size=2,
    ),
    LoRAExperiment(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        experiment_name="llama32-1b",
        rank=16, alpha=32, epochs=3, batch_size=4,
    ),
    LoRAExperiment(
        model_name="google/gemma-3-1b-it",
        experiment_name="gemma3-1b",
        rank=16, alpha=32, epochs=3, batch_size=4,
    ),
]
