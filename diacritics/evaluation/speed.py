"""Inference speed benchmarking for diacritic restoration models."""

import time
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def benchmark_speed(
    predict_fn: Callable[[str], str],
    inputs: list[str],
    warmup: int = 5,
    label: str = "model",
) -> dict[str, float]:
    """Benchmark inference speed of a prediction function.

    Args:
        predict_fn: Takes a string input, returns the restored string.
        inputs: List of test inputs.
        warmup: Number of warmup iterations (not counted).
        label: Name for logging.

    Returns:
        Dict with sentences/sec, chars/sec, total_time_sec, mean_latency_ms.
    """
    for inp in inputs[:warmup]:
        predict_fn(inp)

    total_chars = sum(len(s) for s in inputs)
    start = time.perf_counter()
    for inp in inputs:
        predict_fn(inp)
    elapsed = time.perf_counter() - start

    sents_per_sec = len(inputs) / elapsed
    chars_per_sec = total_chars / elapsed
    mean_latency_ms = (elapsed / len(inputs)) * 1000

    logger.info(
        "%s: %.1f sent/s, %.0f char/s, %.1f ms/sent (total %.1fs on %d items)",
        label, sents_per_sec, chars_per_sec, mean_latency_ms, elapsed, len(inputs),
    )

    return {
        "sentences_per_sec": round(sents_per_sec, 2),
        "chars_per_sec": round(chars_per_sec, 0),
        "total_time_sec": round(elapsed, 2),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "n_items": len(inputs),
        "n_chars": total_chars,
    }
