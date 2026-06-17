from __future__ import annotations

import json
import asyncio
import time
from collections.abc import AsyncIterator, Awaitable
from typing import Any

from .engine import FusionResult, completion_id


def chat_completion_response(
    result: FusionResult,
    *,
    model: str,
    include_debug: bool,
) -> dict[str, Any]:
    message = dict(result.message)
    if include_debug:
        message["fusion_debug"] = result.debug

    finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
    return {
        "id": completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.usage.get("prompt_tokens", 0),
            "completion_tokens": result.usage.get("completion_tokens", 0),
            "total_tokens": result.usage.get("total_tokens", 0),
        },
        "fusion": {
            "elapsed_ms": result.elapsed_ms,
            "degraded": result.degraded,
            "optimized": result.optimized,
            "errors": result.errors or [],
        },
    }


async def delayed_stream_response(
    result: FusionResult,
    *,
    model: str,
) -> AsyncIterator[str]:
    stream_id = completion_id()
    created = int(time.time())

    yield _sse(
        {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
    )

    message = result.message
    content = message.get("content")
    if content:
        for piece in _split_text(content):
            yield _sse(
                {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }
                    ],
                }
            )

    for index, tool_call in enumerate(message.get("tool_calls") or []):
        yield _sse(
            {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": tool_call.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": (tool_call.get("function") or {}).get("name"),
                                        "arguments": (tool_call.get("function") or {}).get(
                                            "arguments", ""
                                        ),
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

    finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
    yield _sse(
        {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": finish_reason}
            ],
        }
    )
    yield "data: [DONE]\n\n"


async def delayed_stream_response_with_heartbeat(
    result_future: Awaitable[FusionResult],
    *,
    model: str,
    heartbeat_seconds: float,
) -> AsyncIterator[str]:
    task = asyncio.ensure_future(result_future)
    interval = max(0.1, heartbeat_seconds)

    yield _sse_comment("fusion-start")
    try:
        while not task.done():
            try:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=interval)
                break
            except asyncio.TimeoutError:
                yield _sse_comment("fusion-keep-alive")
        else:
            result = await task

        async for chunk in delayed_stream_response(result, model=model):
            yield chunk
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def _sse(data: dict[str, Any]) -> str:
    return "data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n\n"


def _sse_comment(comment: str) -> str:
    return f": {comment}\n\n"


def _split_text(text: str, size: int = 80) -> list[str]:
    if len(text) <= size:
        return [text]
    return [text[index : index + size] for index in range(0, len(text), size)]
