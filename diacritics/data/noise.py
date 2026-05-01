"""Noise injection for robustness experiments on Romanian text."""

import random
import string
from dataclasses import dataclass
from .strip import CEDILLA_TO_COMMA, ROMANIAN_DIACRITICS


COMMA_TO_CEDILLA = {
    "\u0219": "\u015F",  # ș -> ş
    "\u0218": "\u015E",  # Ș -> Ş
    "\u021B": "\u0163",  # ț -> ţ
    "\u021A": "\u0162",  # Ț -> Ţ
}

DIACRITIC_SWAPS = {
    "ă": "a", "â": "a", "î": "i", "ș": "s", "ț": "t",
    "Ă": "A", "Â": "A", "Î": "I", "Ș": "S", "Ț": "T",
    "a": "ă", "i": "î", "s": "ș", "t": "ț",
    "A": "Ă", "I": "Î", "S": "Ș", "T": "Ț",
}

KEYBOARD_NEIGHBORS = {
    "a": "sqwz", "s": "adwz", "d": "sfec", "f": "dgrc",
    "i": "uojk", "o": "iplk", "t": "ryfg", "e": "rwds",
}


@dataclass
class NoiseConfig:
    """Configuration for noise injection levels."""
    typo_rate: float = 0.0
    case_flip_rate: float = 0.0
    diacritic_drop_rate: float = 0.0
    diacritic_insert_rate: float = 0.0
    cedilla_mix_rate: float = 0.0


NOISE_LEVELS = {
    "clean": NoiseConfig(),
    "low": NoiseConfig(
        typo_rate=0.01,
        case_flip_rate=0.005,
        diacritic_drop_rate=0.05,
        cedilla_mix_rate=0.1,
    ),
    "medium": NoiseConfig(
        typo_rate=0.03,
        case_flip_rate=0.01,
        diacritic_drop_rate=0.15,
        diacritic_insert_rate=0.02,
        cedilla_mix_rate=0.3,
    ),
    "high": NoiseConfig(
        typo_rate=0.05,
        case_flip_rate=0.02,
        diacritic_drop_rate=0.30,
        diacritic_insert_rate=0.05,
        cedilla_mix_rate=0.5,
    ),
}


def inject_noise(text: str, config: NoiseConfig, rng: random.Random | None = None) -> str:
    """Apply noise to Romanian text according to the given configuration.

    Noise types applied in order:
    1. Cedilla/comma mixing (ș↔ş, ț↔ţ)
    2. Diacritic dropping (ă→a, î→i, etc.)
    3. Diacritic insertion (a→ă, s→ș on non-diacritized chars)
    4. Case flips
    5. Typos (keyboard-neighbor substitution)
    """
    if rng is None:
        rng = random.Random()

    chars = list(text)

    for i, c in enumerate(chars):
        if c in COMMA_TO_CEDILLA and rng.random() < config.cedilla_mix_rate:
            chars[i] = COMMA_TO_CEDILLA[c]

    for i, c in enumerate(chars):
        if c in ROMANIAN_DIACRITICS and rng.random() < config.diacritic_drop_rate:
            chars[i] = DIACRITIC_SWAPS.get(c, c)

    for i, c in enumerate(chars):
        if c not in ROMANIAN_DIACRITICS and c in DIACRITIC_SWAPS:
            if rng.random() < config.diacritic_insert_rate:
                chars[i] = DIACRITIC_SWAPS[c]

    for i, c in enumerate(chars):
        if c.isalpha() and rng.random() < config.case_flip_rate:
            chars[i] = c.swapcase()

    for i, c in enumerate(chars):
        lower = c.lower()
        if lower in KEYBOARD_NEIGHBORS and rng.random() < config.typo_rate:
            neighbors = KEYBOARD_NEIGHBORS[lower]
            replacement = rng.choice(neighbors)
            chars[i] = replacement.upper() if c.isupper() else replacement

    return "".join(chars)


def generate_noisy_variant(
    text: str, level: str = "medium", seed: int | None = None
) -> str:
    """Generate a noisy variant of the text at a named noise level."""
    config = NOISE_LEVELS.get(level)
    if config is None:
        raise ValueError(f"Unknown noise level: {level}. Available: {list(NOISE_LEVELS)}")
    rng = random.Random(seed) if seed is not None else random.Random()
    return inject_noise(text, config, rng)
