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
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None

    def trajectory(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": self.role or "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.finish_reason:
            message["finish_reason"] = self.finish_reason
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
        headers = self._headers(content_type="application/json")

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
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise UpstreamError(f"Malformed upstream response: {data}") from exc
        finish_reason = choice.get("finish_reason")

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
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=data.get("usage"),
        )

    async def model_card(self) -> dict[str, Any] | None:
        base_url = self._api_base_url(self.model_config.url)
        headers = self._headers()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            direct = await self._get_json(
                client,
                f"{base_url}/models/{self.model_config.model_name}",
                headers=headers,
            )
            if self._is_model_card(direct):
                return direct

            listing = await self._get_json(
                client,
                f"{base_url}/models",
                headers=headers,
            )
            if isinstance(listing, dict):
                models = listing.get("data")
                if isinstance(models, list):
                    for item in models:
                        if (
                            self._is_model_card(item)
                            and item.get("id") == self.model_config.model_name
                        ):
                            return item
        return None

    def _chat_completions_url(self, url: str) -> str:
        stripped = url.rstrip("/")
        if stripped.endswith("/chat/completions"):
            return stripped
        return stripped + "/chat/completions"

    def _api_base_url(self, url: str) -> str:
        stripped = url.rstrip("/")
        for suffix in ("/chat/completions", "/responses", "/completions"):
            if stripped.endswith(suffix):
                return stripped[: -len(suffix)]
        return stripped

    def _headers(self, *, content_type: str | None = None) -> dict[str, str]:
        headers = dict(self.model_config.extra_headers)
        if content_type:
            headers["Content-Type"] = content_type
        if self.model_config.api_key:
            headers["Authorization"] = f"Bearer {self.model_config.api_key}"
        if self.model_config.organization:
            headers["OpenAI-Organization"] = self.model_config.organization
        return headers

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        headers: dict[str, str],
    ) -> dict[str, Any] | None:
        try:
            response = await client.get(url, headers=headers)
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def _is_model_card(self, data: Any) -> bool:
        return (
            isinstance(data, dict)
            and isinstance(data.get("id"), str)
            and data.get("object") != "error"
            and "error" not in data
        )

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
