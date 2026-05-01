#!/usr/bin/env python3
"""Evaluate model predictions against gold standard for diacritic restoration."""

import json
import logging
from pathlib import Path

import click

from diacritics.evaluation.metrics import (
    evaluate_batch, aggregate_scores, ALL_METRICS, PRIMARY_METRICS,
)
from diacritics.evaluation.per_char import per_char_scores, format_per_char_report
from diacritics.evaluation.error_analysis import (
    analyze_errors, ai_confusion_rates, format_error_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@click.command()
@click.argument("predictions", type=click.Path(exists=True))
@click.argument("gold", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output JSON file for results (default: stdout)")
@click.option("--primary-only", is_flag=True, help="Report only primary metrics")
@click.option("--error-report", is_flag=True, help="Include detailed error analysis")
@click.option("--per-char", is_flag=True, help="Include per-character F-scores")
def main(predictions: str, gold: str, output: str | None,
         primary_only: bool, error_report: bool, per_char: bool):
    """Evaluate PREDICTIONS against GOLD for diacritic restoration.

    Both files should be JSONL with 'target' field (gold) and 'prediction' field (predictions).
    The gold file uses 'target' for the reference text; the predictions file uses 'prediction'
    for the model output.
    """
    gold_data = load_jsonl(Path(gold))
    pred_data = load_jsonl(Path(predictions))

    gold_map = {r["id"]: r["target"] for r in gold_data}

    golds, preds = [], []
    for rec in pred_data:
        rid = rec["id"]
        if rid not in gold_map:
            logger.warning("Prediction ID %s not in gold set, skipping", rid)
            continue
        golds.append(gold_map[rid])
        preds.append(rec["prediction"])

    logger.info("Evaluating %d pairs", len(golds))

    metrics = {k: v for k, v in ALL_METRICS.items()
               if not primary_only or k in PRIMARY_METRICS}
    per_item = evaluate_batch(golds, preds, metrics)
    aggregated = aggregate_scores(per_item)

    result = {"metrics": aggregated, "n_evaluated": len(golds)}

    if per_char:
        char_scores = per_char_scores(golds, preds)
        result["per_char"] = {
            c: {"precision": s.precision, "recall": s.recall, "f1": s.f1,
                "tp": s.tp, "fp": s.fp, "fn": s.fn}
            for c, s in char_scores.items()
        }

    if error_report:
        stats = analyze_errors(golds, preds)
        ai_conf = ai_confusion_rates(golds, preds)
        result["error_analysis"] = {
            "total_errors": stats.total_errors,
            "under_generation": stats.under_generation,
            "over_generation": stats.over_generation,
            "wrong_diacritic": stats.wrong_diacritic,
            "length_mismatches": stats.length_mismatches,
            "position_errors": stats.position_errors,
            "ai_confusion": ai_conf,
        }

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Results written to %s", out_path)
    else:
        print("\n## Results\n")
        for name, vals in aggregated.items():
            ci = f" [{vals['ci_low']:.4f}, {vals['ci_high']:.4f}]"
            print(f"  {name:15s}: {vals['mean']:.4f}{ci}")

        if per_char and "per_char" in result:
            print("\n## Per-Character Scores\n")
            char_scores_obj = per_char_scores(golds, preds)
            print(format_per_char_report(char_scores_obj))

        if error_report and "error_analysis" in result:
            stats = analyze_errors(golds, preds)
            print(f"\n{format_error_report(stats)}")
            if ai_conf := ai_confusion_rates(golds, preds):
                print("\n### â/î Confusion")
                for k, v in sorted(ai_conf.items()):
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
