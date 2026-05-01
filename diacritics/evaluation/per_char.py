"""Per-character precision/recall/F1 for Romanian diacritic restoration."""

from collections import defaultdict
from dataclasses import dataclass

from diacritics.data.strip import normalize_cedilla

DIACRITIC_CHARS = list("ăâîșț")
DIACRITIC_PAIRS = {
    "ă": "a", "â": "a", "î": "i", "ș": "s", "ț": "t",
    "Ă": "A", "Â": "A", "Î": "I", "Ș": "S", "Ț": "T",
}
ALL_TRACKED = set("ăâîșțĂÂÎȘȚ")


@dataclass
class CharScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def per_char_scores(
    golds: list[str], preds: list[str]
) -> dict[str, CharScore]:
    """Compute per-diacritic-character precision/recall/F1 across a batch.

    For each diacritic character (ă, â, î, ș, ț and uppercase variants):
    - TP: gold has the diacritic AND pred has the same diacritic at that position
    - FP: pred has the diacritic but gold does not (at that position)
    - FN: gold has the diacritic but pred does not (at that position)

    Lowercase and uppercase variants are tracked together (ă+Ă, etc.).
    """
    scores: dict[str, CharScore] = {}
    for c in DIACRITIC_CHARS:
        scores[c] = CharScore()

    for gold, pred in zip(golds, preds):
        gold = normalize_cedilla(gold.strip())
        pred = normalize_cedilla(pred.strip())
        min_len = min(len(gold), len(pred))

        for i in range(min_len):
            gc, pc = gold[i], pred[i]
            gc_lower = gc.lower()
            pc_lower = pc.lower()

            if gc_lower in DIACRITIC_CHARS:
                if pc_lower == gc_lower:
                    scores[gc_lower].tp += 1
                else:
                    scores[gc_lower].fn += 1

            if pc_lower in DIACRITIC_CHARS and pc_lower != gc_lower:
                scores[pc_lower].fp += 1

        for i in range(min_len, len(gold)):
            gc_lower = gold[i].lower()
            if gc_lower in DIACRITIC_CHARS:
                scores[gc_lower].fn += 1

        for i in range(min_len, len(pred)):
            pc_lower = pred[i].lower()
            if pc_lower in DIACRITIC_CHARS:
                scores[pc_lower].fp += 1

    return scores


def confusion_matrix(
    golds: list[str], preds: list[str]
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix for diacritic substitutions.

    Tracks what the model predicts at positions where gold expects a
    diacritizable character (a/ă/â, i/î, s/ș, t/ț).

    Returns: {gold_char: {pred_char: count}}
    """
    tracked = set("aăâiîsștțAĂÂIÎSȘTȚ")
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for gold, pred in zip(golds, preds):
        gold = normalize_cedilla(gold.strip())
        pred = normalize_cedilla(pred.strip())
        min_len = min(len(gold), len(pred))

        for i in range(min_len):
            gc, pc = gold[i].lower(), pred[i].lower()
            if gc in tracked or pc in tracked:
                matrix[gc][pc] += 1

    return {k: dict(v) for k, v in matrix.items()}


def format_per_char_report(scores: dict[str, CharScore]) -> str:
    """Format per-character scores as a markdown table."""
    lines = ["| Char | TP | FP | FN | Precision | Recall | F1 |",
             "|------|----|----|-----|-----------|--------|-----|"]
    for char, s in sorted(scores.items()):
        lines.append(
            f"| {char} | {s.tp} | {s.fp} | {s.fn} | "
            f"{s.precision:.4f} | {s.recall:.4f} | {s.f1:.4f} |"
        )
    return "\n".join(lines)
