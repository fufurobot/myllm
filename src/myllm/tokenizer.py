"""Language-aware tokenization.

Two strategies, chosen per language:

* **Word mode** - for languages that use the space as a delimiter. Text is
  split on whitespace.
* **Character mode** - for languages that do not (CJK, Thai, Khmer, ...). Text
  is tokenized by UTF-8 encoding, i.e. one token per Unicode codepoint.

Which strategy applies is decided from the eBible ``translations.csv``
``script`` column, falling back to a space-ratio heuristic when metadata is
unavailable.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# Scripts that do not delimit words with spaces. Languages written in these
# scripts are tokenized per-character.
NON_SPACING_SCRIPTS: frozenset[str] = frozenset(
    {
        "CJK",
        "Japanese",
        "Thai",
        "Khmer",
        "Lao",
        "Myanmar",
        "Tibetan",
        "Mongolian",
        "Yi",
        "Tai Le",
        "New Tai Lue",
    }
)

# Special token ids. Kept at the bottom of the vocabulary range so that real
# tokens can be numbered from 0 upward.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)

# Number of leading ids reserved for special tokens.
NUM_SPECIAL = len(SPECIAL_TOKENS)


@dataclass(frozen=True)
class LanguageProfile:
    """Resolved tokenization strategy for one language."""

    language_code: str
    script: str
    uses_spaces: bool

    @property
    def mode(self) -> str:
        return "word" if self.uses_spaces else "char"


def load_language_profiles(metadata_csv: Path) -> dict[str, LanguageProfile]:
    """Read ``translations.csv`` and map language code -> tokenization profile.

    When a language has several translations that disagree on script, the most
    conservative choice wins: if *any* entry for the language uses a
    non-spacing script, the language is treated as non-spacing.
    """
    profiles: dict[str, LanguageProfile] = {}
    if not metadata_csv.is_file():
        return profiles

    with metadata_csv.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("languageCode") or "").strip()
            if not code:
                continue
            script = (row.get("script") or "").strip()
            uses_spaces = script not in NON_SPACING_SCRIPTS

            existing = profiles.get(code)
            if existing is None:
                profiles[code] = LanguageProfile(code, script, uses_spaces)
            elif existing.uses_spaces and not uses_spaces:
                # Prefer the non-spacing interpretation.
                profiles[code] = LanguageProfile(code, script, False)

    return profiles


def infer_profile_from_text(text: str, language_code: str = "??") -> LanguageProfile:
    """Guess the tokenization mode from a text sample.

    Used when the language is absent from the metadata. The rule: if spaces are
    essentially never present between characters, the script does not delimit
    words, so tokenize per character. A small threshold is used rather than an
    exact zero because some non-spacing scripts still contain occasional
    spaces (e.g. between sentences).
    """
    sample = text[:200_000]
    if not sample:
        return LanguageProfile(language_code, "unknown", True)

    n_chars = len(sample)
    n_spaces = sum(1 for ch in sample if ch.isspace())
    ratio = n_spaces / max(n_chars, 1)

    # Latin-ish text sits around 0.15-0.20; CJK/Thai fall below 0.01.
    uses_spaces = ratio >= 0.02
    return LanguageProfile(
        language_code, "unknown", uses_spaces
    )


def parse_language_code(path: Path) -> str:
    """Extract the language code from an eBible filename (``eng-engfbv.txt``)."""
    return path.name.split("-", 1)[0]


# Word mode keeps letters/digits and internal apostrophes/hyphens so that
# "don't" and "well-known" stay single tokens.
_WORD_RE = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)


class Tokenizer:
    """A vocabulary plus encode/decode for one language.

    The vocabulary is built from the training text. Ids ``0..NUM_SPECIAL-1``
    are reserved for the special tokens; learned tokens start after them.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        mode: str,
        language_code: str = "??",
    ) -> None:
        if mode not in {"word", "char"}:
            raise ValueError(f"mode must be 'word' or 'char', got {mode!r}")
        self.mode = mode
        self.language_code = language_code
        self.id_to_token: list[str] = list(vocab)
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(vocab)}

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    # -- construction --------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        mode: str,
        min_freq: int = 1,
        max_vocab: int | None = None,
        language_code: str = "??",
    ) -> Tokenizer:
        """Build a vocabulary from ``texts``.

        Args:
            texts: Iterable of text samples.
            mode: ``"word"`` or ``"char"``.
            min_freq: Drop tokens seen fewer than this many times.
            max_vocab: Keep at most this many learned tokens (most frequent).
            language_code: Stored for reference/debugging.
        """
        from collections import Counter

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(cls._split(text, mode))

        most_common = counts.most_common()
        if min_freq > 1:
            most_common = [(t, c) for t, c in most_common if c >= min_freq]
        if max_vocab is not None:
            most_common = most_common[:max_vocab]

        vocab = list(SPECIAL_TOKENS) + [t for t, _ in most_common]
        return cls(vocab, mode, language_code)

    @staticmethod
    def _split(text: str, mode: str) -> list[str]:
        """Split raw text into token strings (no id mapping)."""
        if mode == "char":
            # One token per Unicode codepoint. Normalizing to NFC keeps
            # canonically equivalent sequences from splitting the vocabulary.
            return [ch for ch in unicodedata.normalize("NFC", text) if not ch.isspace()]
        return _WORD_RE.findall(text.lower())

    # -- encoding ------------------------------------------------------------

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to ids, mapping unknown tokens to ``<unk>``."""
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in self._split(text, self.mode)]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_many(
        self,
        texts: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> list[list[int]]:
        """Encode each line independently (useful when lines are sentences)."""
        out: list[list[int]] = []
        for text in texts:
            ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            if ids:
                out.append(ids)
        return out

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Decode ids back to text.

        In word mode tokens are joined by spaces; in char mode they are
        concatenated. This is lossy (case folding in word mode), so it is
        intended for inspection rather than exact round-tripping.
        """
        special = set(range(NUM_SPECIAL))
        toks: list[str] = []
        for i in ids:
            if not 0 <= i < len(self.id_to_token):
                continue
            if skip_special and i in special:
                continue
            toks.append(self.id_to_token[i])
        sep = " " if self.mode == "word" else ""
        return sep.join(toks)

    def iter_special_ids(self) -> Iterator[int]:
        yield from range(NUM_SPECIAL)

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist the vocabulary as JSON."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": self.mode,
            "language_code": self.language_code,
            "vocab": self.id_to_token,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        """Load a vocabulary written by :meth:`save`."""
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["vocab"], payload["mode"], payload.get("language_code", "??"))
