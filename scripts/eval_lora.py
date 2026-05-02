#!/usr/bin/env python3
"""Evaluate LoRA models using MLX batch_generate for fast inference."""

import json
import logging
import time
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


PROMPT_TEMPLATE = "Restore diacritics: {input}\n"


@click.command()
@click.option("--model", required=True, help="Base model HF ID")
@click.option("--adapter-path", required=True, help="Path to LoRA adapters dir")
@click.option("--data-dir", default="data/splits", help="Data splits directory")
@click.option("--output-dir", required=True, help="Output directory for results")
@click.option("--name", required=True, help="Model name for logging")
@click.option("--batch-size", default=32, help="Continuous batching size")
def main(model, adapter_path, data_dir, output_dir, name, batch_size):
    from mlx_lm import load, batch_generate
    from diacritics.evaluation.metrics import evaluate_batch, aggregate_scores
    from diacritics.evaluation.per_char import per_char_scores, format_per_char_report
    from diacritics.evaluation.error_analysis import analyze_errors, ai_confusion_rates, format_error_report

    data = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s with adapter %s", model, adapter_path)
    mlx_model, tokenizer = load(model, adapter_path=adapter_path)

    def predict_batch(texts):
        prompts = [tokenizer.encode(PROMPT_TEMPLATE.format(input=t)) for t in texts]
        max_tokens = max(len(t) for t in texts) + 50
        resp = batch_generate(
            mlx_model, tokenizer,
            prompts=prompts,
            max_tokens=max_tokens,
            completion_batch_size=batch_size,
        )
        results = []
        for text in resp.texts:
            clean = text.split("\n")[0].strip()
            results.append(clean)
        return results

    all_results = {}

    for test_file in sorted(data.glob("test_*.jsonl")):
        test_name = test_file.stem
        test_data = load_jsonl(test_file)
        inputs = [r["input"] for r in test_data]
        golds = [r["target"] for r in test_data]

        logger.info("Evaluating %s (%d items, batch_size=%d)", test_name, len(inputs), batch_size)
        start = time.time()
        preds = predict_batch(inputs)
        elapsed = time.time() - start
        logger.info("%s: %.1fs (%.1f items/s)", test_name, elapsed, len(inputs) / elapsed)

        per_item = evaluate_batch(golds, preds)
        aggregated = aggregate_scores(per_item)
        all_results[test_name] = aggregated
        logger.info("%s: RA_CS_WL=%.4f, DER=%.4f",
                    test_name, aggregated["RA_CS_WL"]["mean"], aggregated["DER"]["mean"])

        pred_records = [{"id": r["id"], "prediction": p} for r, p in zip(test_data, preds)]
        with open(out / f"predictions_{test_name}.jsonl", "w", encoding="utf-8") as f:
            for rec in pred_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    clean_test = next((f for f in data.glob("test_*clean*") if "crawler" in f.stem), None)
    if clean_test:
        test_data = load_jsonl(clean_test)
        golds = [r["target"] for r in test_data]
        preds_clean = [json.loads(l)["prediction"] for l in open(out / f"predictions_{clean_test.stem}.jsonl")]

        cs = per_char_scores(golds, preds_clean)
        with open(out / "per_char_report.md", "w") as f:
            f.write(format_per_char_report(cs))

        es = analyze_errors(golds, preds_clean)
        ai = ai_confusion_rates(golds, preds_clean)
        with open(out / "error_report.md", "w") as f:
            f.write(format_error_report(es))
            if ai:
                f.write("\n\n### a/i Confusion\n")
                for k, v in sorted(ai.items()):
                    f.write(f"- {k}: {v}\n")

    # Speed: single-item for fair comparison with other models
    from mlx_lm import generate
    def predict_single(text):
        prompt = PROMPT_TEMPLATE.format(input=text)
        result = generate(mlx_model, tokenizer, prompt=prompt, max_tokens=len(text) + 50)
        return result.split("\n")[0].strip()

    from diacritics.evaluation.speed import benchmark_speed
    test_data = load_jsonl(clean_test)
    speed = benchmark_speed(predict_single, [r["input"] for r in test_data][:100], warmup=3, label=name)
    all_results["speed"] = speed

    with open(out / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("All results saved to %s", out)


if __name__ == "__main__":
    main()
