#!/usr/bin/env python3
"""Train and evaluate the dictionary (most-frequent-form) baseline."""

import json
import logging
from pathlib import Path

import click

from diacritics.models.dictionary import DictionaryBaseline
from diacritics.evaluation.metrics import evaluate_batch, aggregate_scores, PRIMARY_METRICS, ALL_METRICS
from diacritics.evaluation.per_char import per_char_scores, format_per_char_report
from diacritics.evaluation.error_analysis import analyze_errors, ai_confusion_rates, format_error_report
from diacritics.evaluation.speed import benchmark_speed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@click.command()
@click.option("--data-dir", type=click.Path(exists=True), default="data/splits")
@click.option("--output-dir", type=click.Path(), default="artifacts/dictionary")
def main(data_dir: str, output_dir: str):
    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_data = load_jsonl(data / "train.jsonl")
    pairs = [(r["input"], r["target"]) for r in train_data]
    logger.info("Training dictionary on %d pairs", len(pairs))

    model = DictionaryBaseline()
    model.train(pairs)
    model.save(out / "dictionary.json")

    test_files = sorted(data.glob("test_*.jsonl"))
    all_results = {}

    for test_file in test_files:
        test_name = test_file.stem
        test_data = load_jsonl(test_file)
        inputs = [r["input"] for r in test_data]
        golds = [r["target"] for r in test_data]

        preds = [model.predict(inp) for inp in inputs]

        per_item = evaluate_batch(golds, preds)
        aggregated = aggregate_scores(per_item)

        pred_records = [{"id": r["id"], "prediction": p} for r, p in zip(test_data, preds)]
        pred_path = out / f"predictions_{test_name}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for rec in pred_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        all_results[test_name] = aggregated
        logger.info("%s: RA_CS_WL=%.4f, DER=%.4f",
                    test_name,
                    aggregated["RA_CS_WL"]["mean"],
                    aggregated["DER"]["mean"])

    if any("clean" in name for name in all_results):
        clean_test = next(f for f in test_files if "clean" in f.stem)
        test_data = load_jsonl(clean_test)
        inputs = [r["input"] for r in test_data]
        golds = [r["target"] for r in test_data]
        preds = [model.predict(inp) for inp in inputs]

        char_scores = per_char_scores(golds, preds)
        err_stats = analyze_errors(golds, preds)
        ai_conf = ai_confusion_rates(golds, preds)

        speed = benchmark_speed(model.predict, inputs, warmup=5, label="dictionary")
        all_results["speed"] = speed

        with open(out / "per_char_report.md", "w") as f:
            f.write(format_per_char_report(char_scores))
        with open(out / "error_report.md", "w") as f:
            f.write(format_error_report(err_stats))
            if ai_conf:
                f.write("\n\n### â/î Confusion\n")
                for k, v in sorted(ai_conf.items()):
                    f.write(f"- {k}: {v}\n")

    with open(out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("All results saved to %s/results.json", out)


if __name__ == "__main__":
    main()
