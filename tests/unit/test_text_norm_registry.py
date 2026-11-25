"""MOCK/unit tests: no model weights are loaded."""
import pytest
from ml.preprocessing.text_norm import normalize_text
from ml.inference import registry

def test_numbers_expanded_per_language():
    assert "three" in normalize_text("I have 3 apples", "en")
    assert "tres" in normalize_text("Tengo 3 manzanas", "es")
    assert "drei" in normalize_text("Ich habe 3 Äpfel", "de")

def test_native_scripts_preserved():
    assert normalize_text("नमस्ते,  दुनिया", "hi") == "नमस्ते, दुनिया"
    assert normalize_text("안녕하세요", "ko") == "안녕하세요"

def test_accents_kept_and_punctuation_stripped():
    assert normalize_text("¡Hola, señor!", "es") == "¡hola señor"

def test_language_matrix_matches_verified_checkpoints():
    m = registry.language_matrix()
    assert "mms-tts" not in m["ja"]["models"] and "mms-tts" not in m["it"]["models"]
    assert "xtts-v2" in m["ja"]["models"] and "xtts-v2" in m["it"]["models"]
    assert "mms-tts" in m["hi"]["models"]
    assert all(v["models"] for v in m.values())

def test_validate_rejects_bad_combos():
    with pytest.raises(ValueError): registry.validate("mms-tts", "ja")
    with pytest.raises(ValueError): registry.validate("xtts-v2", "en", emotion="happy")
    with pytest.raises(ValueError): registry.validate("mms-tts", "en", speaking_rate=3)
    with pytest.raises(KeyError): registry.validate("nope", "en")

def test_hindi_numbers_exact():
    from ml.preprocessing.hindi_numbers import hindi_words
    assert hindi_words(25) == "पच्चीस" and hindi_words(100) == "एक सौ"
    assert hindi_words(123456) == "एक लाख तेईस हज़ार चार सौ छप्पन"

def test_numbers_expanded_ar_ko_ja_hi():
    assert "25" not in normalize_text("لدي 25 تفاحة", "ar")
    assert normalize_text("사과 3개", "ko") == "사과 삼 개"
    assert "三" in normalize_text("3個", "ja")
    assert "पच्चीस" in normalize_text("25 सेब", "hi")

def test_f5_registered_en_only_requires_reference():
    b = registry.get_backend("f5-tts")
    assert b.supported_languages() == ["en"] and b.supported_features()["requires_reference"]
    with pytest.raises(ValueError): b.synthesize(__import__("ml.models.base", fromlist=["x"]).SynthesisRequest("hi", "en"))

def test_italian_lowercased_and_accents_kept():
    assert normalize_text("Buongiorno, è già Perché", "it") == "buongiorno è già perché"

def test_multilingual_tokenizer_shared_vocab_covers_all_targets():
    from ml.preprocessing.tokenizer import multilingual_tokenizer
    t = multilingual_tokenizer()
    samples = {"en": "Hello world", "hi": "नमस्ते दुनिया", "ar": "مرحبا بالعالم", "ko": "안녕하세요", "ja": "こんにちは", "es": "¿Cómo estás?",
               "de": "Schöne Grüße", "fr": "Où est l'été", "pt": "Olá, coração", "it": "Buongiorno a tutti"}
    for lang, txt in samples.items():
        prepared = t.prepare(txt, lang); ids = t.encode(txt, lang, add_blank=False)
        assert ids and len(ids) == len(prepared), (lang, prepared)       # nothing silently dropped
        assert all(i < t.vocab_size for i in ids)
    assert t.prepare("नमस्ते", "hi").isascii()                             # romanized into the shared Latin vocabulary

def test_english_tokenizer_unchanged_by_multilingual_support():
    from ml.preprocessing.tokenizer import english_tokenizer
    t = english_tokenizer(); assert t.vocab_size == 34 and t.encode("Hi!", "en")[1::2] == [t.idx["h"], t.idx["i"]]
