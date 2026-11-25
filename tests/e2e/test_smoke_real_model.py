"""REAL model inference smoke test (downloads MMS weights on first run). Not mocked."""
import time, pytest, soundfile as sf
from fastapi.testclient import TestClient

pytestmark = pytest.mark.real_model

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from backend.app.config import settings
    settings.data_dir = tmp_path_factory.mktemp("data")
    from backend.app.main import app
    with TestClient(app) as c: yield c

def _wait(c, jid, timeout=180):
    t = time.time()
    while time.time() - t < timeout:
        j = c.get(f"/api/v1/jobs/{jid}").json()
        if j["status"] in ("succeeded", "failed"): return j
        time.sleep(0.5)
    raise TimeoutError

def test_generate_wav_and_mp3(client, tmp_path):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    for fmt in ("wav", "mp3"):
        jid = client.post("/api/v1/tts/generate", json={"text": "Hello from PolyVoice.", "language": "en", "format": fmt}).json()["job_id"]
        j = _wait(client, jid); assert j["status"] == "succeeded", j["error"]
        r = client.get(f"/api/v1/audio/{j['audio_id']}")
        assert r.status_code == 200 and r.headers["x-ai-generated"] == "true"
        p = tmp_path / f"o.{fmt}"; p.write_bytes(r.content)
        info = sf.info(p); assert info.duration > 0.8 and info.samplerate == 16000 and info.channels == 1

def test_invalid_language_rejected(client):
    r = client.post("/api/v1/tts/generate", json={"text": "x", "language": "ja"})
    assert r.status_code == 400 and "does not support" in r.json()["error"]["message"]

def test_emotion_rejected_on_unsupported_model(client):
    r = client.post("/api/v1/tts/generate", json={"text": "x", "language": "en", "emotion": "happy"})
    assert r.status_code == 400
