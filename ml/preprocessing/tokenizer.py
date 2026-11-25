"""Character tokenizer with blank interspersing (VITS add_blank). Language-aware via per-language normalization."""
from __future__ import annotations
import json, re, unicodedata
from pathlib import Path
from ml.preprocessing.text_norm import normalize_text

PAD = "_"

class CharTokenizer:
    def __init__(self, symbols: list[str]):
        self.symbols = [PAD] + [s for s in symbols if s != PAD]
        self.idx = {s: i for i, s in enumerate(self.symbols)}
    @property
    def vocab_size(self): return len(self.symbols)
    def prepare(self, text: str, language: str) -> str:
        """Normalize, then romanize non-Latin scripts (uroman, as MMS does) so all languages share one Latin-script vocabulary."""
        if language == "zh":
            return normalize_text(_zh_to_pinyin(text), "zh")
        t = normalize_text(text, language)
        if language in ROMANIZE_LANGUAGES:
            t = _uroman().romanize_string(t).lower()
            t = re.sub(r"\s+", " ", re.sub(r"[^a-z' ,.?!-]", " ", unicodedata.normalize("NFKD", t))).strip()
        return t
    def encode(self, text: str, language: str = "en", add_blank: bool = True) -> list[int]:
        ids = [self.idx[c] for c in self.prepare(text, language) if c in self.idx]
        if not add_blank: return ids
        out = [0] * (len(ids) * 2 + 1)
        out[1::2] = ids
        return out
    def save(self, path): Path(path).write_text(json.dumps({"symbols": self.symbols}, ensure_ascii=False))
    @classmethod
    def load(cls, path): return cls(json.loads(Path(path).read_text())["symbols"])

ENGLISH_SYMBOLS = list(" ,.?!'-abcdefghijklmnopqrstuvwxyz")
def english_tokenizer() -> CharTokenizer: return CharTokenizer(ENGLISH_SYMBOLS)

ROMANIZE_LANGUAGES = {"hi", "ar", "ko", "ja", "zh", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"}   # scripts that are not Latin; romanization is lossy (esp. Arabic vowels, Japanese kanji)
_UR = None
def _uroman():
    global _UR
    if _UR is None:
        import uroman; _UR = uroman.Uroman()
    return _UR

MULTILINGUAL_SYMBOLS = list(" ,.?!'-abcdefghijklmnopqrstuvwxyz\u00e1\u00e0\u00e2\u00e3\u00e4\u00e7\u00e9\u00e8\u00ea\u00eb\u00ed\u00ee\u00ef\u00f1\u00f3\u00f4\u00f5\u00f6\u00fa\u00f9\u00fb\u00fc\u00df\u00e6\u0153\u00bf\u00a1")
def multilingual_tokenizer() -> CharTokenizer:
    """Shared character vocabulary for the 10 target languages (Latin-script text + romanized hi/ar/ko/ja)."""
    return CharTokenizer(MULTILINGUAL_SYMBOLS)

_CJK = re.compile(r"[\u3400-\u9fff]")
def _zh_to_pinyin(text: str) -> str:
    """Hanzi -> toned pinyin ('guang3 zhou1', neutral tone = 5) with pypinyin when installed, else toneless uroman. Text that is already ASCII pinyin
    (as stored in the AISHELL-3 manifest) is returned unchanged. Non-CJK characters are kept for normalize_text to filter."""
    if not _CJK.search(text): return text
    try:
        from pypinyin import pinyin, Style
        syl = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors=lambda s: [s])
        return " ".join(x[0] for x in syl)
    except ImportError:
        return _uroman().romanize_string(text)

# Superset covering de fr it es nl en (accented Latin), toned pinyin digits (zh) and romanized ar/Indic.
ALL_SYMBOLS = MULTILINGUAL_SYMBOLS + list("0123456789") + [c for c in "\u00e5\u00ec\u00f2\u00e1\u00ef\u00ff" if c not in MULTILINGUAL_SYMBOLS]
def all_languages_tokenizer() -> CharTokenizer:
    return CharTokenizer(list(dict.fromkeys(ALL_SYMBOLS)))
