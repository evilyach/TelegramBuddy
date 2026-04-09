from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource

TTSDevice = Literal["cpu", "mps", "cuda"]


class CharacterConfig(BaseModel):
    tg_bot_token: str
    bot_id: str
    name: str           # primary display name (e.g. "Чикеряу"); checked for mentions
    aliases: list[str] = []  # extra trigger words (e.g. ["Чикер", "Чик", "Мистер Ч"])
    ref_audio: Path
    ref_text: str
    prompt: str  # inline text or path to a file
    answer_threshold: int = 6  # minimum intent score (0-10) to respond
    voice_word_count_threshold: int = 15
    tts_denoise: bool = True
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
    tts_device: TTSDevice = "cpu"  # "mps" for Mac, "cuda" for GPU server
    characters: dict[str, CharacterConfig]

    model_config = SettingsConfigDict(yaml_file="config.yaml")

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (YamlConfigSettingsSource(settings_cls),)


def load_config(path: str = "config.yaml") -> AppConfig:
    return AppConfig(_yaml_file=path)
