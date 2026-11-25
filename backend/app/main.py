import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from backend.app.config import settings
from backend.app.database import db
from backend.app.database.db import Job, Audio, session
from backend.app.schemas.tts import TTSRequest
from backend.app.services import tts_service
from ml.inference import registry, voices as voice_catalog
from ml.inference.device import detect_device

@asynccontextmanager
async def lifespan(app):
    db.init_db(); yield

app = FastAPI(title="PolyVoice AI", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                   allow_headers=["*"], expose_headers=["X-Request-ID", "X-AI-Generated"])

@app.middleware("http")
async def trace(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    resp = await call_next(request); resp.headers["X-Request-ID"] = rid; return resp

def _err(status, code, msg, details=None):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": msg, "details": details}})

@app.exception_handler(RequestValidationError)
async def _val(_, exc): return _err(422, "validation_error", "Invalid request", exc.errors().__repr__())

@app.exception_handler(HTTPException)
async def _http(_, exc): return _err(exc.status_code, "http_error", str(exc.detail))

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "device": str(detect_device())}

@app.get("/api/v1/models")
def models(): return {"models": registry.list_models()}

@app.get("/api/v1/voices")
def voices(language: str | None = None, model: str | None = None):
    return {"voices": voice_catalog.list_voices(language, model)}

@app.get("/api/v1/voices/{voice_id}/sample")
def voice_sample(voice_id: str):
    v = voice_catalog.get_voice(voice_id)  # filename comes from the manifest, never from the URL
    if not v or not v.get("sample"): raise HTTPException(404, "No sample for this voice")
    f = voice_catalog.SAMPLES / v["sample"]
    if not f.exists(): raise HTTPException(404, "Sample file missing; run scripts/build_voice_manifest.py")
    return FileResponse(f, media_type="audio/wav", headers={"X-AI-Generated": "true"})

@app.get("/api/v1/languages")
def languages(): return {"languages": registry.language_matrix()}

@app.post("/api/v1/tts/generate", status_code=202)
def generate(req: TTSRequest):
    if len(req.text) > settings.max_text_chars:
        raise HTTPException(413, f"Text exceeds {settings.max_text_chars} characters")
    try:
        registry.validate(req.model, req.language, req.speaking_rate, req.emotion)
    except KeyError as e: raise HTTPException(404, str(e))
    except ValueError as e: raise HTTPException(400, str(e))
    backend = registry.get_backend(req.model)
    ready, why = backend.is_available()
    if not ready: raise HTTPException(503, f"Model '{req.model}' is not available: {why}")
    if backend.supported_features().get("requires_reference"):
        raise HTTPException(400, f"Model '{req.model}' needs a reference recording; use an enrolled voice (voice enrollment is not available yet)")
    if req.voice_id:
        v = voice_catalog.get_voice(req.voice_id)
        if not v: raise HTTPException(404, f"Unknown voice '{req.voice_id}'")
        if v["model"] != req.model or req.language not in v["languages"]:
            raise HTTPException(400, f"Voice '{req.voice_id}' is not compatible with model '{req.model}' and language '{req.language}'")
    return {"job_id": tts_service.submit(req.model_dump())}

@app.get("/api/v1/jobs")
def list_jobs(limit: int = 20):
    with session() as s:
        rows = s.query(Job).order_by(Job.created_at.desc()).limit(min(limit, 100)).all()
    return {"jobs": [_job(j) for j in rows]}

def _job(j: Job):
    return {"id": j.id, "status": j.status, "audio_id": j.audio_id, "error": j.error,
            "metrics": j.metrics, "request": j.request, "created_at": j.created_at.isoformat()}

@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str):
    with session() as s: j = s.get(Job, job_id)
    if not j: raise HTTPException(404, "Job not found")
    return _job(j)

@app.get("/api/v1/audio/{audio_id}")
def get_audio(audio_id: str):
    # IDs are 32-hex random; path comes from the DB, never from the URL.
    with session() as s: a = s.get(Audio, audio_id)
    if not a: raise HTTPException(404, "Audio not found")
    mt = "audio/mpeg" if a.path.endswith(".mp3") else "audio/wav"
    return FileResponse(a.path, media_type=mt, headers={"X-AI-Generated": "true"},
                        filename=f"polyvoice-{audio_id[:8]}.{a.path.rsplit('.',1)[-1]}")
