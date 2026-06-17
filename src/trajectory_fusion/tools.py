from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any


REASONING_FIELD_NAMES = {
    "reasoning",
    "reasoning_content",
    "reasoning_text",
    "thinking",
    "thought",
    "thoughts",
}
REASONING_TAG_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


def available_tool_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function":
            function = tool.get("function") or {}
            name = function.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def normalize_tool_calls(
    tool_calls: Any,
    *,
    allowed_names: list[str],
) -> list[dict[str, Any]] | None:
    if not tool_calls:
        return None
    if not isinstance(tool_calls, list):
        return None

    normalized: list[dict[str, Any]] = []
    allowed = set(allowed_names)
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or (allowed and name not in allowed):
            continue
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        if not isinstance(arguments, str):
            arguments = "{}"
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            parsed = {}
        normalized.append(
            {
                "id": call.get("id") or f"call_fusion_{index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(parsed, ensure_ascii=False),
                },
            }
        )
    return normalized or None


def strip_reasoning_text(content: Any) -> str | None:
    if content is None:
        return None
    if not isinstance(content, str):
        content = str(content)
    content = REASONING_TAG_PATTERN.sub("", content)
    stripped = content.strip()
    return stripped or None


def sanitize_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {
        "role": trajectory.get("role") or "assistant",
        "content": strip_reasoning_text(trajectory.get("content")),
    }
    tool_calls = trajectory.get("tool_calls")
    if tool_calls:
        sanitized["tool_calls"] = deepcopy(tool_calls)
    for key, value in trajectory.items():
        if key in sanitized or key in REASONING_FIELD_NAMES:
            continue
        sanitized[key] = deepcopy(value)
    return sanitized


def normalize_trajectory(
    trajectory: dict[str, Any],
    *,
    allowed_tool_names: list[str],
    primary_trajectory: dict[str, Any],
) -> dict[str, Any]:
    content = strip_reasoning_text(trajectory.get("content"))

    tool_calls = normalize_tool_calls(
        trajectory.get("tool_calls"),
        allowed_names=allowed_tool_names,
    )

    if allowed_tool_names and trajectory.get("tool_calls") and not tool_calls:
        tool_calls = normalize_tool_calls(
            primary_trajectory.get("tool_calls"),
            allowed_names=allowed_tool_names,
        )

    normalized: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    return normalized


def apply_hybrid_judge_update(
    primary_trajectory: dict[str, Any],
    content_decision: dict[str, Any],
    *,
    allowed_tool_names: list[str],
    judge_tool_calls: Any = None,
) -> dict[str, Any]:
    message = normalize_trajectory(
        deepcopy(primary_trajectory),
        allowed_tool_names=allowed_tool_names,
        primary_trajectory=primary_trajectory,
    )

    if content_decision.get("operation") == "replace":
        message["content"] = strip_reasoning_text(content_decision.get("text"))

    if judge_tool_calls:
        tool_calls = normalize_tool_calls(
            judge_tool_calls,
            allowed_names=allowed_tool_names,
        )
        if tool_calls:
            message["tool_calls"] = tool_calls

    return message


apply_trajectory_update = apply_hybrid_judge_update
apply_trajectory_patch = apply_hybrid_judge_update
