from pydantic import BaseModel, Field

class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = Field(pattern=r"^[a-z]{2,3}$")
    model: str = "mms-tts"
    voice_id: str | None = None
    speaking_rate: float = Field(1.0, ge=0.5, le=2.0)
    emotion: str | None = None
    format: str = Field("wav", pattern="^(wav|mp3)$")
