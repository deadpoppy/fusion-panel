from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2)
            return os.environ.get(name, default or "")

        return ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8082
    request_timeout_seconds: float = 180
    public_base_url: str | None = None
    model_name: str = "glus"
    client_api_key: str = "fusion-panel"


class FusionConfig(BaseModel):
    enabled_by_default: bool = True
    delayed_streaming: bool = True
    stream_heartbeat_seconds: float = 10
    include_debug_in_response: bool = False
    expose_debug_headers: bool = True
    debug_dump_dir: str | None = None
    record_dir: str | None = None
    max_panel_concurrency: int = 8
    panel_timeout_seconds: float = 90
    judge_timeout_seconds: float = 90
    aux_timeout_primary_multiplier: float = 2.0
    fallback_to_primary_on_failure: bool = True


class BaseModelConfig(BaseModel):
    name: str | None = None
    url: str
    api_key: str = ""
    model_name: str
    organization: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.name or self.model_name


class PrimaryModelConfig(BaseModelConfig):
    pass


class ModelConfig(BaseModelConfig):
    temperature: float | None = None


class ModelsConfig(BaseModel):
    primary: PrimaryModelConfig
    aux: list[ModelConfig] = Field(default_factory=list)
    judge: ModelConfig


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    models: ModelsConfig


def load_config(path: str | Path | None = None) -> AppConfig:
    load_dotenv()
    config_path = Path(path or os.environ.get("FUSION_CONFIG", "config.yaml"))
    if not config_path.exists():
        example = Path("config.yaml.example")
        hint = f" Copy {example} to {config_path} and edit model url/api_key/model_name values." if example.exists() else ""
        raise FileNotFoundError(f"Config file not found: {config_path}.{hint}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    expanded = expand_env(raw)
    return AppConfig.model_validate(expanded)
