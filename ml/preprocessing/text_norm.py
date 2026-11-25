"""Language-aware text normalization. Numerals are expanded with num2words where it supports the language."""
from __future__ import annotations
import re, unicodedata

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_POINT = {"en": "point", "es": "coma", "fr": "virgule", "de": "komma", "pt": "vírgula", "hi": "दशमलव", "ar": "فاصلة", "ko": "점", "ja": "点", "it": "virgola", "nl": "komma"}
_N2W = {"en": "en", "hi": "hi", "es": "es", "fr": "fr", "de": "de", "ar": "ar", "ko": "ko",
        "pt": "pt", "it": "it", "ja": "ja", "nl": "nl"}
# Characters kept per language (after lowercasing where case exists); others become spaces.
_KEEP = {
    "en": r"a-z' ", "es": r"a-záéíóúüñ¿¡' ", "fr": r"a-zàâæçéèêëîïôœùûüÿ' ",
    "it": r"a-zàèéìíîòóùú' ", "de": r"a-zäöüß' ", "pt": r"a-záàâãçéêíóôõú' ",
    "nl": r"a-zäëïöüáéíóúèêôç' ",
    "zh": r"a-z0-9 ",   # toned pinyin (tone digit 1-5 per syllable); numerals are NOT expanded for zh
}

def _expand_numbers(text: str, lang: str) -> str:
    """Spell out digits. Decimals use '.' as separator; thousands separators are not guessed."""
    if lang == "hi":
        from ml.preprocessing.hindi_numbers import hindi_words
        def words(n): return hindi_words(n)
    elif lang in _N2W:
        try: from num2words import num2words
        except ImportError: return text
        def words(n): return num2words(n, lang=_N2W[lang])
    else:
        return text
    def sub(m):
        raw = m.group(0)
        try:
            if re.search(r"[.]", raw):
                whole, frac = raw.split(".", 1)
                return " " + words(int(whole)) + " " + _POINT.get(lang, ".") + " " + " ".join(words(int(d)) for d in frac) + " "
            return " " + words(int(raw)) + " "
        except Exception:
            return raw
    return _NUM.sub(sub, text)

def normalize_text(text: str, language: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _expand_numbers(text, language)
    if language in _KEEP:
        text = text.lower()
        text = re.sub(rf"[^{_KEEP[language]}]", " ", text)
    return _WS.sub(" ", text).strip()
