#!/usr/bin/env python3
"""Prepare train/val/test splits from the dexonline corpus for diacritic restoration."""

import json
import hashlib
import logging
import random
from pathlib import Path

import click

from diacritics.data.dexonline import load_full_corpus, load_test_sets, DexonlineEntry
from diacritics.data.strip import make_training_pair, normalize_cedilla, is_likely_romanian
from diacritics.data.noise import generate_noisy_variant, NOISE_LEVELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def entry_to_record(entry: DexonlineEntry) -> dict:
    """Convert a DexonlineEntry to a JSONL record with stripped/target pair."""
    target = normalize_cedilla(entry.text)
    source, _ = make_training_pair(entry.text)
    return {
        "id": entry.id,
        "source": entry.source,
        "input": source,
        "target": target,
    }


def deduplicate(
    train_entries: list[DexonlineEntry],
    test_entries: list[DexonlineEntry],
) -> list[DexonlineEntry]:
    """Remove training entries that appear in test sets (by text content)."""
    test_hashes = set()
    for entry in test_entries:
        normalized = normalize_cedilla(entry.text.strip().lower())
        test_hashes.add(hashlib.md5(normalized.encode()).hexdigest())

    original_count = len(train_entries)
    filtered = []
    for entry in train_entries:
        normalized = normalize_cedilla(entry.text.strip().lower())
        h = hashlib.md5(normalized.encode()).hexdigest()
        if h not in test_hashes:
            filtered.append(entry)

    removed = original_count - len(filtered)
    logger.info("Deduplication: removed %d/%d training entries matching test sets",
                removed, original_count)
    return filtered


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


@click.command()
@click.option("--output-dir", type=click.Path(), default="data/splits",
              help="Output directory for JSONL files")
@click.option("--val-ratio", type=float, default=0.1, help="Validation split ratio")
@click.option("--seed", type=int, default=42, help="Random seed for splitting")
@click.option("--noise-seeds", type=int, default=42,
              help="Base seed for noise generation (incremented per level)")
@click.option("--filter-non-romanian/--no-filter", default=True,
              help="Filter entries that don't appear to be Romanian")
def main(output_dir: str, val_ratio: float, seed: int, noise_seeds: int,
         filter_non_romanian: bool):
    out = Path(output_dir)

    test_sets = load_test_sets()
    all_test_entries = []
    for name, entries in test_sets.items():
        all_test_entries.extend(entries)

    full_corpus = load_full_corpus()

    test_ids = {(e.source, e.id) for e in all_test_entries}
    train_pool = [e for e in full_corpus if (e.source, e.id) not in test_ids]
    logger.info("After ID-based test removal: %d training candidates", len(train_pool))

    train_pool = deduplicate(train_pool, all_test_entries)

    if filter_non_romanian:
        before = len(train_pool)
        train_pool = [e for e in train_pool if is_likely_romanian(e.text)]
        logger.info("Romanian filter: kept %d/%d entries", len(train_pool), before)

    rng = random.Random(seed)
    rng.shuffle(train_pool)

    val_size = int(len(train_pool) * val_ratio)
    val_entries = train_pool[:val_size]
    train_entries = train_pool[val_size:]

    logger.info("Split: train=%d, val=%d", len(train_entries), len(val_entries))

    train_records = [entry_to_record(e) for e in train_entries]
    val_records = [entry_to_record(e) for e in val_entries]

    write_jsonl(train_records, out / "train.jsonl")
    write_jsonl(val_records, out / "val.jsonl")

    for name, entries in test_sets.items():
        clean_records = [entry_to_record(e) for e in entries]
        write_jsonl(clean_records, out / f"test_{name}_clean.jsonl")

        for level_name in ["low", "medium", "high"]:
            noisy_records = []
            for i, entry in enumerate(entries):
                target = normalize_cedilla(entry.text)
                noisy_input = generate_noisy_variant(
                    target, level=level_name, seed=noise_seeds + i
                )
                noisy_records.append({
                    "id": entry.id,
                    "source": entry.source,
                    "input": noisy_input,
                    "target": target,
                })
            write_jsonl(noisy_records, out / f"test_{name}_{level_name}.jsonl")

    stats = {
        "train": len(train_records),
        "val": len(val_records),
        "test_crawler_1000": len(test_sets["crawler_1000"]),
        "test_dlrlc_1000": len(test_sets["dlrlc_1000"]),
        "noise_levels": list(NOISE_LEVELS.keys()),
        "seed": seed,
        "val_ratio": val_ratio,
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Dataset stats: %s", stats)


if __name__ == "__main__":
    main()
