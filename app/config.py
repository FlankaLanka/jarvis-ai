"""
Jarvis Voice Assistant - Application Configuration

Loads configuration from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # GitHub Configuration
    github_token: str = Field(default="", env="GITHUB_TOKEN")
    github_default_repo: str = Field(default="", env="GITHUB_DEFAULT_REPO")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Session Configuration
    session_expiry_minutes: int = Field(default=30, env="SESSION_EXPIRY_MINUTES")
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_requests_per_session: int = Field(default=100, env="RATE_LIMIT_REQUESTS_PER_SESSION")
    
    # Audio Configuration
    tts_voice: str = Field(default="alloy", env="TTS_VOICE")
    tts_model: str = Field(default="tts-1", env="TTS_MODEL")
    stt_model: str = Field(default="whisper-1", env="STT_MODEL")
    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    
    # Data Refresh
    api_refresh_interval_seconds: int = Field(default=180, env="API_REFRESH_INTERVAL_SECONDS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

