"""Most-frequent-form dictionary baseline for diacritic restoration.

For each stripped word, predicts the most commonly seen diacritized form
in the training data. Falls back to the input word if unseen.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class DictionaryBaseline:
    """Word-level most-frequent-form lookup."""

    def __init__(self):
        self.word_map: dict[str, str] = {}
        self._counts: dict[str, Counter] = defaultdict(Counter)

    def train(self, pairs: list[tuple[str, str]]):
        """Build lookup from (stripped_input, diacritized_target) pairs."""
        for stripped, target in pairs:
            for sw, tw in zip(stripped.split(), target.split()):
                self._counts[sw.lower()][tw] += 1

        for stripped_word, counter in self._counts.items():
            self.word_map[stripped_word] = counter.most_common(1)[0][0]

        logger.info("Dictionary built: %d unique stripped forms", len(self.word_map))

    def predict(self, text: str) -> str:
        """Restore diacritics word by word using the lookup."""
        words = text.split()
        restored = []
        for word in words:
            lookup_key = word.lower()
            if lookup_key in self.word_map:
                replacement = self.word_map[lookup_key]
                if word[0].isupper() and replacement[0].islower():
                    replacement = replacement[0].upper() + replacement[1:]
                elif word.isupper():
                    replacement = replacement.upper()
                restored.append(replacement)
            else:
                restored.append(word)
        return " ".join(restored)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word_map, f, ensure_ascii=False, indent=0)
        logger.info("Saved dictionary (%d entries) to %s", len(self.word_map), path)

    @classmethod
    def load(cls, path: Path) -> "DictionaryBaseline":
        inst = cls()
        with open(path, encoding="utf-8") as f:
            inst.word_map = json.load(f)
        logger.info("Loaded dictionary (%d entries) from %s", len(inst.word_map), path)
        return inst
