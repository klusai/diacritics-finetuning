"""Diacritic restoration evaluation metrics.

Ports and extends the Summa evaluators with DER, hallucination rate,
bootstrap confidence intervals, and McNemar's test.
"""

import logging
from collections.abc import Sequence

import numpy as np

from diacritics.data.strip import is_diacritizable, DIACRITIZABLE_CHARS, normalize_cedilla

logger = logging.getLogger(__name__)


def _levenshtein(s1, s2) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def restoration_accuracy_char(gold: str, pred: str, case_sensitive: bool = True) -> float:
    """Character-level restoration accuracy. Returns 0 if lengths differ."""
    if not case_sensitive:
        gold, pred = gold.lower(), pred.lower()
    gold, pred = gold.strip(), pred.strip()
    if len(gold) != len(pred):
        return 0.0
    if len(gold) == 0:
        return 1.0
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def restoration_accuracy_word(gold: str, pred: str, case_sensitive: bool = True) -> float:
    """Word-level restoration accuracy. Returns 0 if word counts differ."""
    if not case_sensitive:
        gold, pred = gold.lower(), pred.lower()
    gw, pw = gold.strip().split(), pred.strip().split()
    if len(gw) != len(pw):
        return 0.0
    if len(gw) == 0:
        return 1.0
    return sum(g == p for g, p in zip(gw, pw)) / len(gw)


def restoration_error_rate_char(gold: str, pred: str, case_sensitive: bool = True) -> float:
    """1 - normalized character Levenshtein distance. Higher is better."""
    if not case_sensitive:
        gold, pred = gold.lower(), pred.lower()
    gold, pred = gold.strip(), pred.strip()
    if len(gold) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    return 1.0 - min(_levenshtein(gold, pred) / len(gold), 1.0)


def restoration_error_rate_word(gold: str, pred: str, case_sensitive: bool = True) -> float:
    """1 - normalized word Levenshtein distance. Higher is better."""
    if not case_sensitive:
        gold, pred = gold.lower(), pred.lower()
    gw, pw = gold.strip().split(), pred.strip().split()
    if len(gw) == 0:
        return 1.0 if len(pw) == 0 else 0.0
    return 1.0 - min(_levenshtein(gw, pw) / len(gw), 1.0)


def diacritic_error_rate(gold: str, pred: str) -> float:
    """DER: error rate only at diacritizable positions.

    Diacritizable positions are characters in {a,A,i,I,s,S,t,T} in the
    stripped (base) form that could carry a Romanian diacritic.

    Returns the fraction of diacritizable positions where prediction
    disagrees with gold. Lower is better (unlike the RA/RER metrics).
    Returns 0.0 if no diacritizable positions exist.
    """
    gold, pred = gold.strip(), pred.strip()
    if len(gold) != len(pred):
        diacritizable_count = sum(1 for c in gold if is_diacritizable(c)
                                  or c in "ăâîșțĂÂÎȘȚ")
        return 1.0 if diacritizable_count > 0 else 0.0

    errors = 0
    total = 0
    for g, p in zip(gold, pred):
        base_g = normalize_cedilla(g)
        if is_diacritizable(base_g) or g in "ăâîșțĂÂÎȘȚ":
            total += 1
            if g != p:
                errors += 1
    return errors / total if total > 0 else 0.0


def hallucination_rate(gold: str, pred: str) -> float:
    """Measure hallucination in generative model output.

    Detects: length mismatches, characters outside the expected Romanian
    alphabet, and structural deviations.
    """
    gold, pred = gold.strip(), pred.strip()
    if len(gold) == 0:
        return 0.0

    length_mismatch = abs(len(pred) - len(gold)) / len(gold)

    allowed = set("abcdefghijklmnopqrstuvwxyzăâîșț"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZĂÂÎȘȚ"
                  "0123456789 .,;:!?-–—'\"()[]{}/@#&*+=%$€£\n\t")
    foreign = sum(1 for c in pred if c not in allowed)
    foreign_rate = foreign / max(len(pred), 1)

    return min(length_mismatch + foreign_rate, 1.0)


ALL_METRICS = {
    "RA_CS_CL": lambda g, p: restoration_accuracy_char(g, p, case_sensitive=True),
    "RA_CI_CL": lambda g, p: restoration_accuracy_char(g, p, case_sensitive=False),
    "RA_CS_WL": lambda g, p: restoration_accuracy_word(g, p, case_sensitive=True),
    "RA_CI_WL": lambda g, p: restoration_accuracy_word(g, p, case_sensitive=False),
    "RER_CS_CL": lambda g, p: restoration_error_rate_char(g, p, case_sensitive=True),
    "RER_CI_CL": lambda g, p: restoration_error_rate_char(g, p, case_sensitive=False),
    "RER_CS_WL": lambda g, p: restoration_error_rate_word(g, p, case_sensitive=True),
    "RER_CI_WL": lambda g, p: restoration_error_rate_word(g, p, case_sensitive=False),
    "DER": diacritic_error_rate,
    "HALLUCINATION": hallucination_rate,
}

PRIMARY_METRICS = ["RA_CS_WL", "RA_CS_CL", "DER", "HALLUCINATION"]


def evaluate_pair(gold: str, pred: str, metrics: dict | None = None) -> dict[str, float]:
    """Evaluate a single (gold, prediction) pair on all or selected metrics."""
    if metrics is None:
        metrics = ALL_METRICS
    return {name: fn(gold, pred) for name, fn in metrics.items()}


def evaluate_batch(
    golds: Sequence[str], preds: Sequence[str], metrics: dict | None = None
) -> dict[str, list[float]]:
    """Evaluate a batch of pairs, returning per-item scores for each metric."""
    if metrics is None:
        metrics = ALL_METRICS
    results = {name: [] for name in metrics}
    for gold, pred in zip(golds, preds):
        for name, fn in metrics.items():
            results[name].append(fn(gold, pred))
    return results


def aggregate_scores(
    per_item: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Compute mean and bootstrap 95% CI for each metric."""
    aggregated = {}
    for name, scores in per_item.items():
        arr = np.array(scores)
        mean = float(np.mean(arr))
        ci_low, ci_high = _bootstrap_ci(arr)
        aggregated[name] = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high, "n": len(scores)}
    return aggregated


def _bootstrap_ci(
    scores: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 42
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))
    alpha = 1 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def mcnemar_test(
    golds: Sequence[str], preds_a: Sequence[str], preds_b: Sequence[str],
    metric_fn=None,
) -> dict:
    """McNemar's test for paired model comparison.

    Counts items where model A is correct and B is wrong (and vice versa)
    at word level, then computes chi-squared statistic and p-value.
    """
    if metric_fn is None:
        metric_fn = lambda g, p: restoration_accuracy_word(g, p, case_sensitive=True) == 1.0

    a_right_b_wrong = 0
    a_wrong_b_right = 0
    for g, pa, pb in zip(golds, preds_a, preds_b):
        a_correct = metric_fn(g, pa)
        b_correct = metric_fn(g, pb)
        if a_correct and not b_correct:
            a_right_b_wrong += 1
        elif not a_correct and b_correct:
            a_wrong_b_right += 1

    n = a_right_b_wrong + a_wrong_b_right
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n_discordant": 0}

    chi2 = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / n
    from scipy import stats
    p_value = float(stats.chi2.sf(chi2, df=1))

    return {
        "chi2": float(chi2),
        "p_value": p_value,
        "n_discordant": n,
        "a_right_b_wrong": a_right_b_wrong,
        "a_wrong_b_right": a_wrong_b_right,
    }
