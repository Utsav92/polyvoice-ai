import datetime as dt, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import soundfile as sf
from backend.app.config import settings
from backend.app.database.db import Job, Audio, session, new_id
from ml.inference import registry, voices as voice_catalog
from ml.models.base import SynthesisRequest

# One worker: avoids loading/running several models at once on a 24 GB machine.
_pool = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()

def audio_dir() -> Path:
    d = settings.data_dir / "audio"; d.mkdir(parents=True, exist_ok=True); return d

def submit(req: dict) -> str:
    with session() as s:
        job = Job(kind="tts", request=req); s.add(job); s.commit(); jid = job.id
    _pool.submit(_run, jid)
    return jid

def _set(jid, **kw):
    with session() as s:
        j = s.get(Job, jid)
        for k, v in kw.items(): setattr(j, k, v)
        s.commit()

def _run(jid: str):
    with session() as s: req = s.get(Job, jid).request
    _set(jid, status="running")
    try:
        backend = registry.validate(req["model"], req["language"], req["speaking_rate"], req.get("emotion"))
        v = voice_catalog.get_voice(req["voice_id"]) if req.get("voice_id") else None
        speaker = v["name"] if v and v["model"] == "xtts-v2" else None
        res = backend.synthesize(SynthesisRequest(req["text"], req["language"], speaker, req["speaking_rate"]))
        aid = new_id()
        wav = audio_dir() / f"{aid}.wav"
        sf.write(wav, res.audio, res.sample_rate, subtype="PCM_16")
        path = wav
        if req.get("format") == "mp3":
            mp3 = audio_dir() / f"{aid}.mp3"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame",
                            "-q:a", "2", "-metadata", "comment=AI-generated audio (PolyVoice)", str(mp3)], check=True)
            path = mp3
        dur = len(res.audio) / res.sample_rate
        with session() as s:
            s.add(Audio(id=aid, path=str(path), sample_rate=res.sample_rate, duration_s=dur,
                        model_id=res.model_id, language=req["language"])); s.commit()
        _set(jid, status="succeeded", audio_id=aid, metrics=res.meta,
             finished_at=dt.datetime.now(dt.timezone.utc))
    except Exception as e:  # recorded on the job, never swallowed
        _set(jid, status="failed", error=f"{type(e).__name__}: {e}",
             finished_at=dt.datetime.now(dt.timezone.utc))
