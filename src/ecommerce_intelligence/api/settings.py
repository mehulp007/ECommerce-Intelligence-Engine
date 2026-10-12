"""API configuration from environment variables or a local .env file."""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ecommerce_intelligence.paths import ProjectPaths


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ProjectPaths.default().root / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    eie_api_key: SecretStr | None = None
    eie_cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    eie_text_search: bool = True
    eie_search_device: str = "cpu"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.eie_cors_origins.split(",") if o.strip()]
