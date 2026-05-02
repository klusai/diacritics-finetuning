#!/usr/bin/env python3
"""Re-run prompting baselines via Ollama for apples-to-apples metric comparison."""

import json
import logging
import time
from pathlib import Path

import click
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

RESTORE_PROMPT = """Restore the Romanian diacritics in the following text. Return ONLY the restored text, nothing else.

Example 1:
Input: Maine, cand rasare soarele, voi manca un mar si voi inghiti apa sifonata.
Output: Mâine, când răsare soarele, voi mânca un măr și voi înghiți apă sifonată.

Example 2:
Input: Aceasta este o propozitie fara diacritice care trebuie restaurata.
Output: Aceasta este o propoziție fără diacritice care trebuie restaurată.

Example 3:
Input: Tanarul a inteles ca viata e frumoasa si ca trebuie sa fie atent la fiecare clipa.
Output: Tânărul a înțeles că viața e frumoasă și că trebuie să fie atent la fiecare clipă.

Input: {input}
Output:"""


def restore_via_ollama(client: OpenAI, model: str, text: str, temperature: float = 0.0) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": RESTORE_PROMPT.format(input=text)}],
            temperature=temperature,
            max_tokens=len(text) * 3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Error with %s: %s", model, e)
        return text


@click.command()
@click.option("--model", required=True, help="Ollama model name (e.g. llama3.1:8b)")
@click.option("--test-file", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--ollama-url", default="http://localhost:11434/v1")
@click.option("--limit", type=int, default=None, help="Limit number of items (for testing)")
def main(model: str, test_file: str, output_dir: str, ollama_url: str, limit: int | None):
    client = OpenAI(base_url=ollama_url, api_key="ollama")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    test_data = [json.loads(l) for l in open(test_file) if l.strip()]
    if limit:
        test_data = test_data[:limit]

    model_safe = model.replace(":", "_").replace("/", "_")
    test_name = Path(test_file).stem

    logger.info("Running %s on %d items from %s", model, len(test_data), test_name)

    predictions = []
    start = time.time()
    for i, item in enumerate(test_data):
        pred = restore_via_ollama(client, model, item["input"])
        predictions.append({"id": item["id"], "prediction": pred})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            logger.info("[%d/%d] %.1f items/min", i + 1, len(test_data), rate * 60)

    elapsed = time.time() - start
    logger.info("Completed %d items in %.1f min (%.1f items/min)",
                len(predictions), elapsed / 60, len(predictions) / elapsed * 60)

    pred_path = out / f"predictions_{model_safe}_{test_name}.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved to %s", pred_path)


if __name__ == "__main__":
    main()
