"""Model capability registry: the single source of truth for what each backend supports."""
from __future__ import annotations
from ml.models.base import TTSBackend

_BACKENDS: dict[str, type[TTSBackend]] = {}
_INSTANCES: dict[str, TTSBackend] = {}

def register(cls: type[TTSBackend]):
    _BACKENDS[cls.model_id] = cls; return cls

def _discover():
    from ml.models.mms_adapter.adapter import MMSBackend
    register(MMSBackend)
    from ml.models.xtts_adapter.adapter import XTTSBackend
    register(XTTSBackend)
    from ml.models.f5_adapter.adapter import F5Backend
    register(F5Backend)
    from ml.models.vits.adapter import VitsBackend
    register(VitsBackend)
    from ml.models.fastspeech2.adapter import FastSpeech2Backend
    register(FastSpeech2Backend)

def list_models() -> list[dict]:
    if not _BACKENDS: _discover()
    out = []
    for mid, cls in _BACKENDS.items():
        b = get_backend(mid)
        out.append({"id": mid, "name": cls.display_name, "license": cls.license,
                    "commercial_use": cls.commercial_use, "languages": b.supported_languages(),
                    "features": b.supported_features(), "available": b.is_available()[0], "unavailable_reason": b.is_available()[1], "resources": b.estimate_resource_usage()})
    return out

def get_backend(model_id: str) -> TTSBackend:
    if not _BACKENDS: _discover()
    if model_id not in _BACKENDS:
        raise KeyError(f"Unknown model '{model_id}'")
    if model_id not in _INSTANCES:
        _INSTANCES[model_id] = _BACKENDS[model_id]()
    return _INSTANCES[model_id]

def validate(model_id: str, language: str, speaking_rate: float = 1.0, emotion: str | None = None):
    b = get_backend(model_id)
    if language not in b.supported_languages():
        raise ValueError(f"Model '{model_id}' does not support language '{language}'. "
                         f"Supported: {b.supported_languages()}")
    if emotion and not b.supported_features().get("emotion"):
        raise ValueError(f"Model '{model_id}' does not support emotion control")
    if not 0.5 <= speaking_rate <= 2.0:
        raise ValueError("speaking_rate must be between 0.5 and 2.0")
    return b

TARGET_LANGUAGES = {"en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
                    "ja": "Japanese", "ar": "Arabic", "ko": "Korean", "pt": "Portuguese", "it": "Italian"}

def language_matrix() -> dict:
    """language -> list of model ids that genuinely support it (derived from adapters, never hand-written)."""
    if not _BACKENDS: _discover()
    out = {code: {"name": name, "models": []} for code, name in TARGET_LANGUAGES.items()}
    for mid in _BACKENDS:
        for l in get_backend(mid).supported_languages():
            out.setdefault(l, {"name": l, "models": []})["models"].append(mid)
    return out
