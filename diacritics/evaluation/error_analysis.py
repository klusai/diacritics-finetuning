"""Error analysis for diacritic restoration: confusion patterns, positions, counts."""

from collections import defaultdict
from dataclasses import dataclass

from diacritics.data.strip import normalize_cedilla, ROMANIAN_DIACRITICS


@dataclass
class ErrorStats:
    total_chars: int = 0
    total_errors: int = 0
    over_generation: int = 0
    under_generation: int = 0
    wrong_diacritic: int = 0
    length_mismatches: int = 0
    position_errors: dict = None

    def __post_init__(self):
        if self.position_errors is None:
            self.position_errors = {"start": 0, "mid": 0, "end": 0}


def analyze_errors(
    golds: list[str], preds: list[str]
) -> ErrorStats:
    """Detailed error analysis across a batch of (gold, prediction) pairs."""
    stats = ErrorStats()

    for gold, pred in zip(golds, preds):
        gold = normalize_cedilla(gold.strip())
        pred = normalize_cedilla(pred.strip())

        if len(gold) != len(pred):
            stats.length_mismatches += 1
            continue

        stats.total_chars += len(gold)
        words = gold.split()
        char_idx = 0

        for word in words:
            word_len = len(word)
            for pos_in_word, (gc, pc) in enumerate(
                zip(gold[char_idx:char_idx + word_len], pred[char_idx:char_idx + word_len])
            ):
                if gc == pc:
                    continue

                stats.total_errors += 1

                gc_is_diac = gc in ROMANIAN_DIACRITICS
                pc_is_diac = pc in ROMANIAN_DIACRITICS

                if gc_is_diac and not pc_is_diac:
                    stats.under_generation += 1
                elif not gc_is_diac and pc_is_diac:
                    stats.over_generation += 1
                elif gc_is_diac and pc_is_diac:
                    stats.wrong_diacritic += 1

                if pos_in_word == 0:
                    stats.position_errors["start"] += 1
                elif pos_in_word == word_len - 1:
                    stats.position_errors["end"] += 1
                else:
                    stats.position_errors["mid"] += 1

            char_idx += word_len + 1  # +1 for space

    return stats


def ai_confusion_rates(
    golds: list[str], preds: list[str]
) -> dict[str, int]:
    """Specifically measure â/î confusion rates (the key finding from the prompting paper).

    Returns counts of:
    - â_predicted_as_î: gold=â, pred=î
    - î_predicted_as_â: gold=î, pred=â
    - â_predicted_as_a: gold=â, pred=a (under-generation)
    - î_predicted_as_i: gold=î, pred=i (under-generation)
    - a_predicted_as_â: gold=a, pred=â (over-generation)
    - i_predicted_as_î: gold=i, pred=î (over-generation)
    """
    counts = defaultdict(int)

    for gold, pred in zip(golds, preds):
        gold = normalize_cedilla(gold.strip())
        pred = normalize_cedilla(pred.strip())
        min_len = min(len(gold), len(pred))

        for i in range(min_len):
            gc, pc = gold[i], pred[i]
            if gc == pc:
                continue

            pair = f"{gc}_predicted_as_{pc}"
            key_pairs = {
                "â_predicted_as_î", "î_predicted_as_â",
                "â_predicted_as_a", "î_predicted_as_i",
                "a_predicted_as_â", "i_predicted_as_î",
                "Â_predicted_as_Î", "Î_predicted_as_Â",
            }
            if pair in key_pairs:
                counts[pair] += 1

    return dict(counts)


def format_error_report(stats: ErrorStats) -> str:
    """Format error statistics as a markdown report."""
    lines = [
        "## Error Analysis Report",
        "",
        f"- Total characters analyzed: {stats.total_chars:,}",
        f"- Total errors: {stats.total_errors:,} ({stats.total_errors/max(stats.total_chars,1)*100:.2f}%)",
        f"- Length mismatches (excluded from detailed analysis): {stats.length_mismatches}",
        "",
        "### Error Types",
        f"- Under-generation (missed diacritic): {stats.under_generation}",
        f"- Over-generation (added wrong diacritic): {stats.over_generation}",
        f"- Wrong diacritic (ă↔â, etc.): {stats.wrong_diacritic}",
        "",
        "### Position Distribution",
        f"- Word-initial: {stats.position_errors['start']}",
        f"- Word-medial: {stats.position_errors['mid']}",
        f"- Word-final: {stats.position_errors['end']}",
    ]
    return "\n".join(lines)
