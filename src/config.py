from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource


class CharacterConfig(BaseModel):
    tg_bot_token: str
    bot_id: str
    ref_audio: Path
    ref_text: str
    prompt: str  # inline text or path to a file
    start_conversation_probability: float = 0.01
    voice_word_count_threshold: int = 15
    openrouter_model: str | None = None

    @field_validator("prompt", mode="before")
    @classmethod
    def resolve_prompt(cls, v: str) -> str:
        p = Path(v)
        if p.exists() and p.is_file():
            return p.read_text()
        return v


class AppConfig(BaseSettings):
    openrouter_api_key: str
    openrouter_model: str = "openai/gpt-4o-mini"
    characters: dict[str, CharacterConfig]

    model_config = SettingsConfigDict(yaml_file="config.yaml")

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (YamlConfigSettingsSource(settings_cls),)


def load_config(path: str = "config.yaml") -> AppConfig:
    return AppConfig(_yaml_file=path)
