from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/polyvoice.db"
    max_text_chars: int = 5000
    cors_origins: list[str] = ["http://localhost:3000"]
    model_config = {"env_prefix": "POLYVOICE_", "env_file": ".env"}

settings = Settings()
