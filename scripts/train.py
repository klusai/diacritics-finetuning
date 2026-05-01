#!/usr/bin/env python3
"""Unified training script for all diacritic restoration model families."""

import json
import logging
import time
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_model(predict_fn, data_dir: Path, output_dir: Path, model_name: str):
    """Run evaluation on all test sets and save results."""
    from diacritics.evaluation.metrics import evaluate_batch, aggregate_scores
    from diacritics.evaluation.per_char import per_char_scores, format_per_char_report
    from diacritics.evaluation.error_analysis import analyze_errors, ai_confusion_rates, format_error_report
    from diacritics.evaluation.speed import benchmark_speed

    test_files = sorted(data_dir.glob("test_*.jsonl"))
    all_results = {}

    for test_file in test_files:
        test_name = test_file.stem
        test_data = load_jsonl(test_file)
        inputs = [r["input"] for r in test_data]
        golds = [r["target"] for r in test_data]

        preds = [predict_fn(inp) for inp in inputs]

        per_item = evaluate_batch(golds, preds)
        aggregated = aggregate_scores(per_item)
        all_results[test_name] = aggregated

        pred_records = [{"id": r["id"], "prediction": p} for r, p in zip(test_data, preds)]
        pred_path = output_dir / f"predictions_{test_name}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for rec in pred_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info("%s | %s: RA_CS_WL=%.4f, DER=%.4f",
                    model_name, test_name,
                    aggregated["RA_CS_WL"]["mean"],
                    aggregated["DER"]["mean"])

    clean_test = next((f for f in test_files if "crawler" in f.stem and "clean" in f.stem), None)
    if clean_test:
        test_data = load_jsonl(clean_test)
        inputs = [r["input"] for r in test_data]
        golds = [r["target"] for r in test_data]
        preds = [predict_fn(inp) for inp in inputs]

        char_scores = per_char_scores(golds, preds)
        with open(output_dir / "per_char_report.md", "w") as f:
            f.write(format_per_char_report(char_scores))

        err_stats = analyze_errors(golds, preds)
        ai_conf = ai_confusion_rates(golds, preds)
        with open(output_dir / "error_report.md", "w") as f:
            f.write(format_error_report(err_stats))
            if ai_conf:
                f.write("\n\n### â/î Confusion\n")
                for k, v in sorted(ai_conf.items()):
                    f.write(f"- {k}: {v}\n")

        speed = benchmark_speed(predict_fn, inputs, warmup=5, label=model_name)
        all_results["speed"] = speed

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


@click.group()
def cli():
    """Train diacritic restoration models."""
    pass


@cli.command()
@click.option("--data-dir", type=click.Path(exists=True), default="data/splits")
@click.option("--output-dir", type=click.Path(), default="artifacts/bilstm")
@click.option("--epochs", type=int, default=10)
@click.option("--hidden-size", type=int, default=256)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=1e-3)
def bilstm(data_dir, output_dir, epochs, hidden_size, batch_size, lr):
    """Train character BiLSTM baseline."""
    from diacritics.models.bilstm import BiLSTMModel
    from diacritics.training.config import BiLSTMConfig

    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = BiLSTMConfig(
        max_epochs=epochs, hidden_size=hidden_size,
        batch_size=batch_size, learning_rate=lr,
    )

    train_data = load_jsonl(data / "train.jsonl")
    val_data = load_jsonl(data / "val.jsonl")
    train_pairs = [(r["input"], r["target"]) for r in train_data]
    val_pairs = [(r["input"], r["target"]) for r in val_data]

    logger.info("Training BiLSTM: %d train, %d val", len(train_pairs), len(val_pairs))

    model = BiLSTMModel(config)
    start = time.time()
    model.train(train_pairs, val_pairs)
    elapsed = time.time() - start
    logger.info("Training completed in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    model.save(out / "model")
    evaluate_model(model.predict, data, out, "bilstm")


@cli.command()
@click.option("--data-dir", type=click.Path(exists=True), default="data/splits")
@click.option("--output-dir", type=click.Path(), default="artifacts/dictionary")
def dictionary(data_dir, output_dir):
    """Train dictionary (most-frequent-form) baseline."""
    from diacritics.models.dictionary import DictionaryBaseline

    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_data = load_jsonl(data / "train.jsonl")
    pairs = [(r["input"], r["target"]) for r in train_data]

    model = DictionaryBaseline()
    model.train(pairs)
    model.save(out / "dictionary.json")
    evaluate_model(model.predict, data, out, "dictionary")


if __name__ == "__main__":
    cli()
