"""Diacritic stripping and cedilla normalization for Romanian text."""

import re
import unicodedata


CEDILLA_TO_COMMA = str.maketrans({
    "\u015F": "\u0219",  # ş -> ș
    "\u015E": "\u0218",  # Ş -> Ș
    "\u0163": "\u021B",  # ţ -> ț
    "\u0162": "\u021A",  # Ţ -> Ț
})

ROMANIAN_DIACRITICS = set("ăâîșțĂÂÎȘȚ")

DIACRITIZABLE_CHARS = {
    "a": {"ă", "â"},
    "A": {"Ă", "Â"},
    "i": {"î"},
    "I": {"Î"},
    "s": {"ș"},
    "S": {"Ș"},
    "t": {"ț"},
    "T": {"Ț"},
}


def normalize_cedilla(text: str) -> str:
    """Replace old cedilla-based ş/ţ with standard comma-below ș/ț."""
    return text.translate(CEDILLA_TO_COMMA)


def strip_diacritics(text: str) -> str:
    """Remove all diacritical marks using Unicode NFKD decomposition.

    This strips ALL combining marks, not just Romanian ones. For Romanian
    ADR training pairs this is correct since the input should be pure ASCII-range.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def has_diacritics(text: str) -> bool:
    """Check if text contains any Romanian diacritics."""
    return bool(ROMANIAN_DIACRITICS & set(text))


def is_diacritizable(char: str) -> bool:
    """Check if a character could potentially carry a Romanian diacritic."""
    return char in DIACRITIZABLE_CHARS


def make_training_pair(text: str) -> tuple[str, str]:
    """Create a (stripped_input, diacritized_target) training pair.

    Applies cedilla normalization to the target before stripping.
    """
    target = normalize_cedilla(text)
    source = strip_diacritics(target)
    return source, target


def is_likely_romanian(text: str) -> bool:
    """Heuristic check that text is predominantly Romanian.

    Flags text with unusual Unicode ranges that suggest foreign-language
    content (etymological notes, French/German quotations in dexonline).
    """
    non_romanian_diacritics = set()
    for char in text:
        if unicodedata.combining(unicodedata.normalize("NFKD", char)[-1:]):
            base = unicodedata.normalize("NFKD", char)[0]
            if char not in ROMANIAN_DIACRITICS and base + char not in {
                "aă", "aâ", "iî", "sș", "tț",
                "AĂ", "AÂ", "IÎ", "SȘ", "TȚ",
            }:
                non_romanian_diacritics.add(char)

    if len(non_romanian_diacritics) > 2:
        return False

    romanian_letter_pattern = re.compile(r"[a-zA-ZăâîșțĂÂÎȘȚ]")
    letters = romanian_letter_pattern.findall(text)
    return len(letters) / max(len(text), 1) > 0.5
