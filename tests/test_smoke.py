from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import httpx
from fastapi import FastAPI, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajectory_fusion.config import AppConfig
from trajectory_fusion.engine import FusionEngine, FusionResult
from trajectory_fusion.json_utils import parse_text_decision
from trajectory_fusion.openai_client import ModelResult
from trajectory_fusion.prompts import JUDGE_SYSTEM_PROMPT, build_judge_messages
from trajectory_fusion.responses import delayed_stream_response_with_heartbeat
from trajectory_fusion.server import (
    build_app,
    client_api_key_from_headers,
    startup_usage_text,
)
from trajectory_fusion.tools import apply_hybrid_judge_update, strip_reasoning_text


def upstream_app() -> FastAPI:
    app = FastAPI()
    calls: dict[str, int] = {}
    app.state.calls = calls

    @app.get("/v1/models/{model_id}")
    async def model(model_id: str) -> dict[str, Any]:
        if model_id == "glm-5.2":
            return {
                "id": "glm-5.2",
                "object": "model",
                "created": 123,
                "owned_by": "primary-owner",
                "context_window": 262144,
                "context_length": 262144,
                "max_context_tokens": 262144,
                "max_output_tokens": 65536,
                "supports_tools": True,
                "supports_reasoning": True,
            }
        return {"error": {"message": "not found"}}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "primary",
                    "object": "model",
                    "created": 123,
                    "owned_by": "primary-owner",
                    "context_window": 262144,
                    "max_output_tokens": 65536,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict[str, Any]:
        payload = await request.json()
        model = payload["model"]
        calls[model] = calls.get(model, 0) + 1
        delay = payload.get("mock_delay_seconds")
        if isinstance(delay, int | float) and delay > 0:
            await asyncio.sleep(delay)
        messages = payload.get("messages") or []
        last = messages[-1]["content"] if messages else ""
        tool_calls = None

        if model == "broken-aux":
            return {
                "id": "mock",
                "object": "chat.completion",
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            }

        if model == "broken-primary":
            return {
                "id": "mock",
                "object": "chat.completion",
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            }

        if model == "judge":
            if payload.get("force_invalid_judge"):
                content = "not a decision"
            elif payload.get("force_judge_tool_call"):
                assert payload.get("tools")
                assert payload.get("tool_choice") == "auto"
                content = (
                    "<text_decision>replace</text_decision>\n"
                    "<text_replacement>\nNeed to read the project config first.\n</text_replacement>"
                )
                tool_calls = [
                    {
                        "id": "call_judge_read",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": "{\"filePath\":\"config.yaml\"}",
                        },
                    }
                ]
            else:
                content = (
                    "<text_decision>replace</text_decision>\n"
                    "<text_replacement>\nEnhanced primary answer.\n</text_replacement>"
                )
        elif model == "tool-primary":
            content = None
            tool_calls = [
                {
                    "id": "call_primary_read",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": "{\"filePath\":\"README.md\"}",
                    },
                }
            ]
        elif model == "empty-primary":
            content = None
        elif model == "tool-aux":
            content = "aux saw a possible config issue"
        elif model == "reasoning-primary":
            content = "<think>hidden primary reasoning</think>\nVisible fallback."
        else:
            content = f"{model} saw: {last[:40]}"

        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": "mock",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    return app


async def main() -> None:
    mock_app = upstream_app()
    transport = httpx.ASGITransport(app=mock_app)

    original_client = httpx.AsyncClient

    def patched_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        kwargs["base_url"] = "http://testserver"
        return original_client(*args, **kwargs)

    httpx.AsyncClient = patched_client  # type: ignore[method-assign]
    try:
        config = AppConfig.model_validate(
            {
                "fusion": {
                    "record_dir": tempfile.mkdtemp(prefix="fusion-records-"),
                    "debug_dump_dir": tempfile.mkdtemp(prefix="fusion-dumps-"),
                },
                "models": {
                    "primary": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "glm-5.2",
                    },
                    "aux": [
                        {
                            "name": "aux",
                            "url": "http://testserver/v1",
                            "api_key": "test",
                            "model_name": "aux",
                            "temperature": 0,
                        }
                    ],
                    "judge": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "judge",
                        "temperature": 0,
                    },
                },
            }
        )
        startup_text = startup_usage_text(config, host="0.0.0.0", port=8082)
        assert "base_url: http://127.0.0.1:8082/v1" in startup_text
        assert "api_key: fusion-panel" in startup_text
        assert "model: glus" in startup_text
        assert client_api_key_from_headers({"authorization": "Bearer fusion-panel"}) == "fusion-panel"
        assert client_api_key_from_headers({"x-api-key": "fusion-panel"}) == "fusion-panel"

        fusion_app = build_app(config)
        fusion_transport = httpx.ASGITransport(app=fusion_app)
        async with original_client(
            transport=fusion_transport,
            base_url="http://fusion",
            headers={"Authorization": "Bearer fusion-panel"},
        ) as client:
            models_response = await client.get("/v1/models")
            assert models_response.status_code == 200
            models = models_response.json()
            assert models["object"] == "list"
            assert models["data"][0]["id"] == "glus"
            assert models["data"][0]["object"] == "model"
            assert models["data"][0]["created"] == 123
            assert models["data"][0]["owned_by"] == "primary-owner"
            assert models["data"][0]["context_window"] == 262144
            assert models["data"][0]["context_length"] == 262144
            assert models["data"][0]["max_context_tokens"] == 262144
            assert models["data"][0]["max_output_tokens"] == 65536
            assert models["data"][0]["supports_tools"] is True
            assert models["data"][0]["supports_reasoning"] is True
            assert "root" not in models["data"][0]
            assert "parent" not in models["data"][0]
            model_response = await client.get("/v1/models/glus")
            assert model_response.status_code == 200
            assert model_response.json()["id"] == "glus"
            missing_model_response = await client.get("/v1/models/fusion-panel")
            assert missing_model_response.status_code == 404

        engine = FusionEngine(config)
        result = await engine.run(
            {
                "messages": [
                    {"role": "user", "content": "hello fusion"}
                ]
            }
        )
        assert result.message["content"] == "Enhanced primary answer."
        assert result.debug["mode"] == "judge_integrated_replacement"
        assert result.debug["judge_decision"]["content"]["operation"] == "replace"
        assert "judge_raw" not in result.debug
        assert result.usage == {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
        assert result.fusion_usage
        assert result.fusion_usage["prompt_tokens"] == 3
        assert result.fusion_usage["completion_tokens"] == 3
        assert result.fusion_usage["total_tokens"] == 6
        assert result.fusion_usage["public_usage_source"] == "primary"
        assert len(result.fusion_usage["fusion_calls"]) == 3
        assert result.optimized is True
        optimized_files = list((Path(config.fusion.record_dir) / "optimized").glob("*.json"))
        assert optimized_files
        optimized_record = json.loads(optimized_files[0].read_text())
        assert "raw_content" not in optimized_record["judge"]
        assert optimized_record["judge"]["decision"]["content"]["text"] == "Enhanced primary answer."
        dump_files = list(Path(config.fusion.debug_dump_dir).glob("*.json"))
        assert dump_files
        debug_record = json.loads(dump_files[0].read_text())
        assert "raw_content" not in debug_record["judge"]
        assert debug_record["judge"]["decision"]["content"]["text"] == "Enhanced primary answer."
        stats = await engine.record_stats()
        assert stats["total_requests"] == 1
        assert stats["fusion_requests"] == 1
        assert stats["fusion_optimized_requests"] == 1
        assert stats["optimization_rate_fusion_percent"] == 100.0

        release_stream = asyncio.Event()

        async def blocked_result() -> FusionResult:
            await release_stream.wait()
            primary = ModelResult(
                role="assistant",
                content="final text",
                tool_calls=None,
                raw_message={},
                raw_response={},
                model="primary",
                url="http://test",
                display_name="primary",
                latency_ms=1,
                usage=None,
            )
            return FusionResult(
                message={"role": "assistant", "content": "final text"},
                debug={},
                primary=primary,
                aux=[],
                judge=None,
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                elapsed_ms=30,
            )

        heartbeat_stream = delayed_stream_response_with_heartbeat(
            blocked_result(),
            model="fusion-primary",
            heartbeat_seconds=0.01,
        )
        assert await anext(heartbeat_stream) == ": fusion-start\n\n"
        assert await anext(heartbeat_stream) == ": fusion-keep-alive\n\n"
        release_stream.set()
        heartbeat_chunks = [chunk async for chunk in heartbeat_stream]
        assert any('"object":"chat.completion.chunk"' in chunk for chunk in heartbeat_chunks)
        assert '"choices":[]' in heartbeat_chunks[-2]
        assert '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}' in heartbeat_chunks[-2]
        assert heartbeat_chunks[-1] == "data: [DONE]\n\n"

        config_all_aux_fail = AppConfig.model_validate(
            {
                "fusion": {
                    "record_dir": config.fusion.record_dir,
                },
                "models": {
                    "primary": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "primary",
                    },
                    "aux": [
                        {
                            "name": "broken",
                            "url": "http://testserver/v1",
                            "api_key": "test",
                            "model_name": "broken-aux",
                            "temperature": 0,
                        }
                    ],
                    "judge": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "judge",
                        "temperature": 0,
                    },
                },
            }
        )
        judge_calls_before = mock_app.state.calls.get("judge", 0)
        fallback_engine = FusionEngine(config_all_aux_fail)
        fallback = await fallback_engine.run(
            {
                "messages": [
                    {"role": "user", "content": "hello fallback"}
                ]
            }
        )
        assert fallback.message["content"] == "primary saw: hello fallback"
        assert fallback.debug["mode"] == "primary_passthrough_no_usable_aux"
        assert fallback.degraded is True
        assert fallback.errors == ["broken: Malformed upstream response: {'id': 'mock', 'object': 'chat.completion', 'choices': [], 'usage': {'prompt_tokens': 1, 'completion_tokens': 0, 'total_tokens': 1}}"]
        assert fallback.optimized is False
        assert mock_app.state.calls.get("judge", 0) == judge_calls_before
        stats = await fallback_engine.record_stats()
        assert stats["total_requests"] == 2
        assert stats["fusion_requests"] == 2
        assert stats["fusion_optimized_requests"] == 1
        assert stats["fusion_no_change_requests"] == 1
        assert stats["optimization_rate_fusion_percent"] == 50.0

        slow_aux_config = AppConfig.model_validate(
            {
                "fusion": {
                    "aux_timeout_primary_multiplier": 2.0,
                },
                "models": {
                    "primary": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "primary",
                        "extra_body": {"mock_delay_seconds": 0.02},
                    },
                    "aux": [
                        {
                            "name": "slow-aux",
                            "url": "http://testserver/v1",
                            "api_key": "test",
                            "model_name": "aux",
                            "temperature": 0,
                            "extra_body": {"mock_delay_seconds": 0.2},
                        }
                    ],
                    "judge": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "judge",
                        "temperature": 0,
                    },
                },
            }
        )
        judge_calls_before = mock_app.state.calls.get("judge", 0)
        slow_aux_result = await FusionEngine(slow_aux_config).run(
            {"messages": [{"role": "user", "content": "slow aux"}]}
        )
        assert slow_aux_result.debug["mode"] == "primary_passthrough_no_usable_aux"
        assert slow_aux_result.degraded is True
        assert slow_aux_result.errors
        assert "slow-aux" in slow_aux_result.errors[0]
        assert "timed out" in slow_aux_result.errors[0]
        assert mock_app.state.calls.get("judge", 0) == judge_calls_before

        config_primary_fail = AppConfig.model_validate(
            {
                "models": {
                    "primary": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "broken-primary",
                    },
                    "aux": [
                        {
                            "name": "aux",
                            "url": "http://testserver/v1",
                            "api_key": "test",
                            "model_name": "aux",
                            "temperature": 0,
                        }
                    ],
                    "judge": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "judge",
                        "temperature": 0,
                    },
                },
            }
        )
        primary_fail = await FusionEngine(config_primary_fail).run(
            {"messages": [{"role": "user", "content": "primary fail"}]}
        )
        assert primary_fail.debug["mode"] == "primary_error_passthrough"
        assert primary_fail.degraded is True
        assert primary_fail.message["content"] is None

        invalid_judge_config = AppConfig.model_validate(
            {
                "models": {
                    "primary": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "primary",
                    },
                    "aux": [
                        {
                            "name": "aux",
                            "url": "http://testserver/v1",
                            "api_key": "test",
                            "model_name": "aux",
                            "temperature": 0,
                        }
                    ],
                    "judge": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "judge",
                        "temperature": 0,
                        "extra_body": {"force_invalid_judge": True},
                    },
                },
            }
        )
        invalid_judge = await FusionEngine(invalid_judge_config).run(
            {"messages": [{"role": "user", "content": "judge invalid"}]}
        )
        assert invalid_judge.message["content"] == "primary saw: judge invalid"
        assert invalid_judge.optimized is False
        assert invalid_judge.degraded is True
        assert invalid_judge.judge_decision is None

        invalid_judge_reasoning_config = AppConfig.model_validate(
            {
                "models": {
                    "primary": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "reasoning-primary",
                    },
                    "aux": [
                        {
                            "name": "aux",
                            "url": "http://testserver/v1",
                            "api_key": "test",
                            "model_name": "aux",
                            "temperature": 0,
                        }
                    ],
                    "judge": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "judge",
                        "temperature": 0,
                        "extra_body": {"force_invalid_judge": True},
                    },
                },
            }
        )
        invalid_judge_reasoning = await FusionEngine(invalid_judge_reasoning_config).run(
            {"messages": [{"role": "user", "content": "judge invalid reasoning"}]}
        )
        assert invalid_judge_reasoning.message["content"] == "Visible fallback."
        assert "hidden primary reasoning" not in invalid_judge_reasoning.message["content"]

        tool_message = apply_hybrid_judge_update(
            {
                "role": "assistant",
                "content": "Primary tool request.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": "{\"filePath\":\"README.md\"}",
                        },
                    }
                ],
            },
            {"operation": "replace", "text": "Need to read README first."},
            allowed_tool_names=["read"],
        )
        assert tool_message["content"] == "Need to read README first."
        assert tool_message["tool_calls"][0]["function"]["name"] == "read"

        replaced_tool_message = apply_hybrid_judge_update(
            {
                "role": "assistant",
                "content": "Primary tool request.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": "{\"filePath\":\"README.md\"}",
                        },
                    }
                ],
            },
            {"operation": "replace", "text": "Need to inspect config first."},
            allowed_tool_names=["read"],
            judge_tool_calls=[
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": {"filePath": "config.yaml"},
                    },
                }
            ],
        )
        assert replaced_tool_message["tool_calls"][0]["id"] == "call_2"
        assert replaced_tool_message["tool_calls"][0]["function"]["arguments"] == '{"filePath": "config.yaml"}'

        reasoning_message = apply_hybrid_judge_update(
            {
                "role": "assistant",
                "content": "<think>private reasoning</think>\nVisible primary.",
            },
            {"operation": "replace", "text": "<think>judge reasoning</think>\nVisible enhanced."},
            allowed_tool_names=[],
        )
        assert reasoning_message["content"] == "Visible enhanced."
        assert strip_reasoning_text("<think>x</think>\nShown") == "Shown"

        judge_messages = build_judge_messages(
            {"messages": [{"role": "user", "content": "review"}]},
            ModelResult(
                role="assistant",
                content="<think>hidden primary</think>\nPrimary visible",
                tool_calls=None,
                raw_message={"reasoning_content": "hidden"},
                raw_response={},
                model="primary",
                url="http://test",
                display_name="primary",
                latency_ms=1,
            ),
            [
                ModelResult(
                    role="assistant",
                    content="<think>hidden aux</think>\nAux visible",
                    tool_calls=None,
                    raw_message={"reasoning": "hidden"},
                    raw_response={},
                    model="aux",
                    url="http://test",
                    display_name="aux",
                    latency_ms=1,
                )
            ],
            [],
        )
        judge_payload_text = judge_messages[-1]["content"]
        assert "hidden primary" not in judge_payload_text
        assert "hidden aux" not in judge_payload_text
        assert "Primary visible" in judge_payload_text
        assert "Aux visible" in judge_payload_text
        assert "current stage of work" in JUDGE_SYSTEM_PROMPT
        assert "non-consensus status visible" in JUDGE_SYSTEM_PROMPT
        assert "How to express five-lens information" in JUDGE_SYSTEM_PROMPT
        assert "same language pattern as PRIMARY" in JUDGE_SYSTEM_PROMPT
        assert "Content decision format" in JUDGE_SYSTEM_PROMPT
        assert "native OpenAI tool calls" in JUDGE_SYSTEM_PROMPT
        assert "Return STRICT JSON only" not in JUDGE_SYSTEM_PROMPT

        parsed = parse_text_decision(
            "<think>hidden</think>\n"
            "<text_decision>replace</text_decision>\n"
            "<text_replacement>\ndone\n</text_replacement>"
        )
        assert parsed == {"operation": "replace", "text": "done"}
        assert parse_text_decision("<text_decision>none</text_decision>") == {
            "operation": "none",
            "text": None,
        }
        assert parse_text_decision("not a decision") == {}

        native_tool_config = AppConfig.model_validate(
            {
                "models": {
                    "primary": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "primary",
                    },
                    "aux": [
                        {
                            "name": "aux",
                            "url": "http://testserver/v1",
                            "api_key": "test",
                            "model_name": "aux",
                            "temperature": 0,
                        }
                    ],
                    "judge": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "judge",
                        "temperature": 0,
                        "extra_body": {"force_judge_tool_call": True},
                    },
                },
            }
        )
        native_tool_result = await FusionEngine(native_tool_config).run(
            {
                "messages": [{"role": "user", "content": "inspect config"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "Read a file",
                            "parameters": {
                                "type": "object",
                                "properties": {"filePath": {"type": "string"}},
                                "required": ["filePath"],
                            },
                        },
                    }
                ],
                "tool_choice": "required",
            }
        )
        assert native_tool_result.message["content"] == "Need to read the project config first."
        assert native_tool_result.message["tool_calls"][0]["id"] == "call_judge_read"
        assert native_tool_result.debug["judge_decision"]["tool_calls"]["operation"] == "replace"

        tool_only_config = AppConfig.model_validate(
            {
                "models": {
                    "primary": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "tool-primary",
                    },
                    "aux": [
                        {
                            "name": "tool-aux",
                            "url": "http://testserver/v1",
                            "api_key": "test",
                            "model_name": "tool-aux",
                            "temperature": 0,
                        }
                    ],
                    "judge": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "judge",
                        "temperature": 0,
                    },
                },
            }
        )
        judge_calls_before = mock_app.state.calls.get("judge", 0)
        tool_only_result = await FusionEngine(tool_only_config).run(
            {
                "messages": [{"role": "user", "content": "read README"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "Read a file",
                            "parameters": {
                                "type": "object",
                                "properties": {"filePath": {"type": "string"}},
                                "required": ["filePath"],
                            },
                        },
                    }
                ],
            }
        )
        assert tool_only_result.debug["mode"] == "primary_passthrough_tool_only"
        assert tool_only_result.message["content"] is None
        assert tool_only_result.message["tool_calls"][0]["id"] == "call_primary_read"
        assert mock_app.state.calls.get("judge", 0) == judge_calls_before

        empty_primary_config = AppConfig.model_validate(
            {
                "models": {
                    "primary": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "empty-primary",
                    },
                    "aux": [
                        {
                            "name": "aux",
                            "url": "http://testserver/v1",
                            "api_key": "test",
                            "model_name": "aux",
                            "temperature": 0,
                        }
                    ],
                    "judge": {
                        "url": "http://testserver/v1",
                        "api_key": "test",
                        "model_name": "judge",
                        "temperature": 0,
                    },
                },
            }
        )
        judge_calls_before = mock_app.state.calls.get("judge", 0)
        empty_primary_result = await FusionEngine(empty_primary_config).run(
            {"messages": [{"role": "user", "content": "empty primary"}]}
        )
        assert empty_primary_result.debug["mode"] == "primary_passthrough_empty"
        assert empty_primary_result.message["content"] is None
        assert "tool_calls" not in empty_primary_result.message
        assert empty_primary_result.optimized is False
        assert mock_app.state.calls.get("judge", 0) == judge_calls_before
        print("smoke ok")
    finally:
        httpx.AsyncClient = original_client  # type: ignore[method-assign]


if __name__ == "__main__":
    asyncio.run(main())
