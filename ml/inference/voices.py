"""Voice catalog. XTTS built-ins come from a manifest generated from the real model (scripts/build_voice_manifest.py)."""
from __future__ import annotations
import json
from pathlib import Path
from ml.inference import registry

_XTTS = Path(__file__).resolve().parents[1] / "models" / "xtts_adapter" / "voices.json"
SAMPLES = Path("data/voice_samples")

def _builtin() -> list[dict]:
    voices = []
    mms_langs = registry.get_backend("mms-tts").supported_languages()
    voices.append({"id": "mms_default", "name": "MMS default", "model": "mms-tts", "languages": mms_langs,
                   "description": "Single fixed voice of each MMS language checkpoint", "style": "neutral",
                   "license": "CC-BY-NC-4.0", "ownership": "Meta MMS checkpoint", "sample": None})
    if _XTTS.exists(): voices += json.loads(_XTTS.read_text())
    return voices

def list_voices(language: str | None = None, model: str | None = None) -> list[dict]:
    out = []
    for v in _builtin():
        if language and language not in v["languages"]: continue
        if model and v["model"] != model: continue
        out.append({**v, "sample_url": f"/api/v1/voices/{v['id']}/sample" if v.get("sample") else None,
                    "supported_models": [v["model"]]})
    return out

def get_voice(voice_id: str) -> dict | None:
    return next((v for v in _builtin() if v["id"] == voice_id), None)
