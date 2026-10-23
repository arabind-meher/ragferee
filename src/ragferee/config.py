"""Application settings."""

from functools import lru_cache
from typing import Literal, Self
from urllib.parse import quote

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    # Postgres DB
    postgres_user: str = Field(min_length=1)
    postgres_password: SecretStr = Field(min_length=1)
    postgres_db: str = "ragferee"
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)

    # LLM
    llm_provider: Literal["mock", "openai"] = "mock"
    openai_api_key: SecretStr | None = None

    # Validators
    @model_validator(mode="after")
    def check_openai_key(self) -> Self:
        """Require an OpenAI API key when the OpenAI provider is selected."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key is required.")
        return self

    def get_postgres_url(self) -> str:
        """Return the Postgres connection URL with the password escaped."""
        return (
            f"postgresql://{self.postgres_user}"
            f":{quote(self.postgres_password.get_secret_value(), safe='')}"
            f"@{self.postgres_host}"
            f":{self.postgres_port}"
            f"/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, loaded once and cached."""
    return Settings()
