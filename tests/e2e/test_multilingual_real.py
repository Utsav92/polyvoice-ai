"""REAL inference across languages (downloads weights on first run; XTTS test needs .venv-xtts)."""
import time, pytest, soundfile as sf
from fastapi.testclient import TestClient
pytestmark = pytest.mark.real_model

TEXT = {"en": "Hello, how are you today?", "hi": "नमस्ते, आज मौसम अच्छा है।", "es": "Hola, ¿cómo estás hoy?",
        "fr": "Bonjour, comment allez-vous ?", "de": "Hallo, wie geht es dir heute?", "ar": "مرحبا، كيف حالك اليوم؟",
        "ko": "안녕하세요, 오늘 어떻게 지내세요?", "pt": "Olá, como você está hoje?"}

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from backend.app.config import settings
    settings.data_dir = tmp_path_factory.mktemp("data")
    from backend.app.main import app
    with TestClient(app) as c: yield c

def _gen(c, tmp, **body):
    jid = c.post("/api/v1/tts/generate", json=body).json()["job_id"]; t = time.time()
    while time.time() - t < 600:
        j = c.get(f"/api/v1/jobs/{jid}").json()
        if j["status"] in ("succeeded", "failed"): break
        time.sleep(0.5)
    assert j["status"] == "succeeded", j["error"]
    p = tmp / "o.wav"; p.write_bytes(c.get(f"/api/v1/audio/{j['audio_id']}").content); return sf.info(p)

@pytest.mark.parametrize("lang", list(TEXT))
def test_mms_language(client, tmp_path, lang):
    info = _gen(client, tmp_path, text=TEXT[lang], language=lang, model="mms-tts")
    assert info.duration > 0.8 and info.channels == 1

@pytest.mark.parametrize("lang,text", [("ja", "こんにちは、お元気ですか。"), ("it", "Ciao, come stai oggi?")])
def test_xtts_language(client, tmp_path, lang, text):
    info = _gen(client, tmp_path, text=text, language=lang, model="xtts-v2")
    assert info.duration > 0.8 and info.samplerate == 24000

def test_no_single_model_covers_unsupported(client):
    assert client.post("/api/v1/tts/generate", json={"text": "x", "language": "ja", "model": "mms-tts"}).status_code == 400

def test_voices_endpoint_and_xtts_voice(client, tmp_path):
    vs = client.get("/api/v1/voices", params={"language": "it", "model": "xtts-v2"}).json()["voices"]
    assert len(vs) == 58 and all(v["license"] for v in vs)
    s = client.get(vs[0]["sample_url"]); assert s.status_code == 200 and s.headers["x-ai-generated"] == "true"
    info = _gen(client, tmp_path, text="Ciao a tutti.", language="it", model="xtts-v2", voice_id=vs[0]["id"])
    assert info.duration > 0.5

def test_voice_model_mismatch_and_f5_without_reference_rejected(client):
    vs = client.get("/api/v1/voices", params={"model": "xtts-v2"}).json()["voices"]
    assert client.post("/api/v1/tts/generate", json={"text": "x", "language": "en", "model": "mms-tts", "voice_id": vs[0]["id"]}).status_code == 400
    assert client.post("/api/v1/tts/generate", json={"text": "x", "language": "en", "model": "f5-tts"}).status_code == 400
    assert client.post("/api/v1/tts/generate", json={"text": "x", "language": "en", "voice_id": "nope"}).status_code == 404

def test_custom_vits_plumbing(client, tmp_path):
    """Plumbing only: proves the trained-checkpoint path produces a valid waveform via the API. Says nothing about intelligibility."""
    from ml.inference import registry
    ok, why = registry.get_backend("polyvoice-vits-lj").is_available()
    if not ok: pytest.skip(why)
    info = _gen(client, tmp_path, text="Hello, this is a test.", language="en", model="polyvoice-vits-lj")
    assert info.duration > 0.5 and info.samplerate == 16000 and info.channels == 1

def test_unavailable_model_returns_503(client, monkeypatch):
    from ml.inference import registry
    b = registry.get_backend("polyvoice-vits-lj")
    monkeypatch.setattr(type(b), "is_available", lambda self: (False, "no checkpoint"))
    r = client.post("/api/v1/tts/generate", json={"text": "x", "language": "en", "model": "polyvoice-vits-lj"})
    assert r.status_code == 503 and "no checkpoint" in r.json()["error"]["message"]
