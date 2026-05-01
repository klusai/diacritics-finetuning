"""Load and manage dexonline corpus data for diacritic restoration experiments."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUMMA_DATA_ROOT = Path(os.environ.get("DEXONLINE_DATA_DIR", "data/dexonline"))

CORPUS_PATHS = {
    "crawler_all": SUMMA_DATA_ROOT / "crawler/json/dexonline_crawler_all.json",
    "crawler_1000": SUMMA_DATA_ROOT / "crawler/json/dexonline_crawler_1000.json",
    "crawler_100": SUMMA_DATA_ROOT / "crawler/json/dexonline_crawler_100.json",
    "crawler_10": SUMMA_DATA_ROOT / "crawler/json/dexonline_crawler_10.json",
    "dlrlc_all": SUMMA_DATA_ROOT / "dlrlc/json/dexonline_dlrlc_all.json",
    "dlrlc_10": SUMMA_DATA_ROOT / "dlrlc/json/dexonline_dlrlc_10.json",
}


@dataclass
class DexonlineEntry:
    id: int
    text: str
    source: str
    metadata: dict


def load_corpus(name: str) -> list[DexonlineEntry]:
    """Load a named dexonline corpus slice.

    Args:
        name: One of the keys in CORPUS_PATHS (e.g. "crawler_all", "dlrlc_all").

    Returns:
        List of DexonlineEntry with the diacritized text and metadata.
    """
    path = CORPUS_PATHS.get(name)
    if path is None:
        raise ValueError(f"Unknown corpus: {name}. Available: {list(CORPUS_PATHS)}")
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    source_type = "crawler" if "crawler" in name else "dlrlc"
    entries = []
    for item in raw:
        meta = {k: v for k, v in item.items() if k not in ("id", "text")}
        entries.append(
            DexonlineEntry(
                id=item["id"],
                text=item["text"],
                source=source_type,
                metadata=meta,
            )
        )

    logger.info("Loaded %d entries from %s (%s)", len(entries), name, path)
    return entries


def load_full_corpus() -> list[DexonlineEntry]:
    """Load the combined CRAWLER + DLRLC corpus (~338k entries)."""
    crawler = load_corpus("crawler_all")
    dlrlc = load_corpus("dlrlc_all")
    combined = crawler + dlrlc
    logger.info("Full corpus: %d entries (crawler=%d, dlrlc=%d)",
                len(combined), len(crawler), len(dlrlc))
    return combined


def load_test_sets() -> dict[str, list[DexonlineEntry]]:
    """Load the canonical test sets matching the published InnoComp paper.

    Returns:
        Dict with keys "crawler_1000" and "dlrlc_1000".
    """
    crawler_test = load_corpus("crawler_1000")

    dlrlc_all = load_corpus("dlrlc_all")
    dlrlc_test = dlrlc_all[:1000]
    for entry in dlrlc_test:
        entry.source = "dlrlc"

    logger.info("Test sets: crawler=%d, dlrlc=%d", len(crawler_test), len(dlrlc_test))
    return {"crawler_1000": crawler_test, "dlrlc_1000": dlrlc_test}
