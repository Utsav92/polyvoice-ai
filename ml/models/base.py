"""Common TTS backend interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SynthesisRequest:
    text: str
    language: str                  # ISO 639-1 code, e.g. "en"
    voice_id: str | None = None
    speaking_rate: float = 1.0
    emotion: str | None = None
    reference_audio: str | None = None
    reference_text: str | None = None      # transcript of reference_audio (required by F5-TTS)

@dataclass
class SynthesisResult:
    audio: np.ndarray              # float32 mono, [-1, 1]
    sample_rate: int
    model_id: str
    meta: dict = field(default_factory=dict)

class TTSBackend(ABC):
    model_id: str
    display_name: str
    license: str
    commercial_use: bool

    def is_available(self) -> tuple[bool, str]:
        """(ready, reason-if-not). Backends override when they need weights/venvs that may be absent."""
        return True, ""

    @abstractmethod
    def load_model(self, language: str | None = None) -> None: ...
    @abstractmethod
    def supported_languages(self) -> list[str]: ...
    @abstractmethod
    def supported_features(self) -> dict[str, bool]: ...
    @abstractmethod
    def synthesize(self, req: SynthesisRequest) -> SynthesisResult: ...
    @abstractmethod
    def estimate_resource_usage(self, language: str | None = None) -> dict: ...
    @abstractmethod
    def release_model(self) -> None: ...
