"""Build the XTTS built-in voice manifest + sample clips from the real model. Gender/age are NOT labeled (unknown)."""
import json, sys
from pathlib import Path
import soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.models.xtts_adapter.adapter import XTTSBackend, XTTS_LANGS
from ml.models.base import SynthesisRequest

OUT = Path("data/voice_samples"); OUT.mkdir(parents=True, exist_ok=True)
b = XTTSBackend(); b.load_model()
manifest = []
for name in b.speakers:
    vid = "xtts_" + name.lower().replace(" ", "_")
    f = OUT / f"{vid}.wav"
    if not f.exists():
        r = b.synthesize(SynthesisRequest("Hello, this is a sample of my voice.", "en", voice_id=name))
        sf.write(f, r.audio, r.sample_rate)
    manifest.append({"id": vid, "name": name, "model": "xtts-v2", "languages": XTTS_LANGS,
                     "description": "Built-in XTTS-v2 speaker (voice attributes not labeled by the model authors)",
                     "style": "unlabeled", "license": "Coqui Public Model License (non-commercial), bundled with XTTS-v2",
                     "ownership": "Bundled with model; provenance per Coqui", "sample": f"{vid}.wav"})
    print(vid, flush=True)
Path("ml/models/xtts_adapter/voices.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
b.release_model()
