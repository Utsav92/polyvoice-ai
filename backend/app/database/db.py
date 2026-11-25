import uuid, datetime as dt
from sqlalchemy import create_engine, String, Float, Integer, Text, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from backend.app.config import settings

def _now(): return dt.datetime.now(dt.timezone.utc)
def new_id(): return uuid.uuid4().hex

class Base(DeclarativeBase): pass

class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(32), default="tts")
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|succeeded|failed
    request: Mapped[dict] = mapped_column(JSON)
    audio_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

class Audio(Base):
    __tablename__ = "audio"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    path: Mapped[str] = mapped_column(String(512))
    sample_rate: Mapped[int] = mapped_column(Integer)
    duration_s: Mapped[float] = mapped_column(Float)
    model_id: Mapped[str] = mapped_column(String(64))
    language: Mapped[str] = mapped_column(String(8))
    ai_generated: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

_engine = None
_Session = None

def init_db(url: str | None = None):
    global _engine, _Session
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(url or settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)

def session():
    return _Session()
