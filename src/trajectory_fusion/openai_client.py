from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import BaseModelConfig


class UpstreamError(RuntimeError):
    pass


@dataclass
class ModelResult:
    role: str
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    raw_message: dict[str, Any]
    raw_response: dict[str, Any]
    model: str
    url: str
    display_name: str
    latency_ms: int
    usage: dict[str, Any] | None = None
    error: str | None = None

    def trajectory(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": self.role or "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


class OpenAICompatibleClient:
    def __init__(
        self,
        model_config: BaseModelConfig,
        *,
        timeout_seconds: float,
    ) -> None:
        self.model_config = model_config
        self.timeout_seconds = timeout_seconds

    async def chat_completion(
        self,
        payload: dict[str, Any],
    ) -> ModelResult:
        url = self._chat_completions_url(self.model_config.url)
        body = self._build_body(self.model_config, payload)
        headers = {
            "Content-Type": "application/json",
            **self.model_config.extra_headers,
        }
        if self.model_config.api_key:
            headers["Authorization"] = f"Bearer {self.model_config.api_key}"
        if self.model_config.organization:
            headers["OpenAI-Organization"] = self.model_config.organization

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=body)
            if response.status_code >= 400 and self._should_retry_without_json_schema(
                response,
                body,
            ):
                body = dict(body)
                body["response_format"] = {"type": "json_object"}
                response = await client.post(url, headers=headers, json=body)
        latency_ms = int((time.perf_counter() - start) * 1000)
        if response.status_code >= 400:
            raise UpstreamError(
                f"{self.model_config.display_name} returned "
                f"{response.status_code}: {response.text[:1000]}"
            )

        data = response.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise UpstreamError(f"Malformed upstream response: {data}") from exc

        return ModelResult(
            role=message.get("role", "assistant"),
            content=message.get("content"),
            tool_calls=message.get("tool_calls"),
            raw_message=message,
            raw_response=data,
            model=self.model_config.model_name,
            url=url,
            display_name=self.model_config.display_name,
            latency_ms=latency_ms,
            usage=data.get("usage"),
        )

    def _chat_completions_url(self, url: str) -> str:
        stripped = url.rstrip("/")
        if stripped.endswith("/chat/completions"):
            return stripped
        return stripped + "/chat/completions"

    def _build_body(self, model_config: BaseModelConfig, payload: dict[str, Any]) -> dict[str, Any]:
        passthrough_keys = {
            "messages",
            "tools",
            "tool_choice",
            "response_format",
            "parallel_tool_calls",
            "seed",
            "stop",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
        }
        body = {key: payload[key] for key in passthrough_keys if key in payload}
        body["model"] = model_config.model_name
        body["stream"] = False
        temperature = getattr(model_config, "temperature", None)
        if temperature is not None:
            body["temperature"] = temperature
        elif "temperature" in payload:
            body["temperature"] = payload["temperature"]
        body.update(model_config.extra_body)
        return body

    def _should_retry_without_json_schema(
        self,
        response: httpx.Response,
        body: dict[str, Any],
    ) -> bool:
        response_format = body.get("response_format")
        if not isinstance(response_format, dict):
            return False
        if response_format.get("type") != "json_schema":
            return False
        text = response.text.lower()
        return (
            "response_format" in text
            or "json_schema" in text
            or "schema" in text
            or response.status_code in {400, 422}
        )
