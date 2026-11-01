from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LLM_PROVIDER: str = "mock"
    LLM_MODEL: str = "mock-model"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()