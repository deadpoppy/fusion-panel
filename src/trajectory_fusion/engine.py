from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig, BaseModelConfig
from .json_utils import parse_text_decision
from .openai_client import ModelResult, OpenAICompatibleClient
from .prompts import build_judge_messages
from .tools import (
    apply_hybrid_judge_update,
    available_tool_names,
    normalize_tool_calls,
    normalize_trajectory,
    sanitize_trajectory,
    strip_reasoning_text,
)


@dataclass
class FusionResult:
    message: dict[str, Any]
    debug: dict[str, Any]
    primary: ModelResult
    aux: list[ModelResult]
    judge: ModelResult | None
    usage: dict[str, Any]
    elapsed_ms: int
    degraded: bool = False
    errors: list[str] | None = None
    optimized: bool = False
    judge_decision: dict[str, Any] | None = None


class FusionEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.fusion.max_panel_concurrency)
        self._record_lock = asyncio.Lock()

    async def run(self, payload: dict[str, Any]) -> FusionResult:
        start = time.perf_counter()
        errors: list[str] = []

        primary_task = asyncio.create_task(
            self._safe_model_call(
                self.config.models.primary,
                payload,
                timeout_seconds=self.config.fusion.panel_timeout_seconds,
            )
        )
        aux_tasks = [
            asyncio.create_task(
                self._safe_model_call(
                    model,
                    payload,
                    timeout_seconds=self.config.fusion.panel_timeout_seconds,
                )
            )
            for model in self.config.models.aux
        ]

        primary = await primary_task
        if primary.error:
            for task in aux_tasks:
                task.cancel()
            if aux_tasks:
                await asyncio.gather(*aux_tasks, return_exceptions=True)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result = FusionResult(
                message=self._primary_passthrough_message(primary, payload),
                debug={
                    "mode": "primary_error_passthrough",
                    "primary_error": primary.error,
                    "skipped_fusion": True,
                    "optimized": False,
                },
                primary=primary,
                aux=[],
                judge=None,
                usage=self._usage(primary, []),
                elapsed_ms=elapsed_ms,
                degraded=True,
                errors=[f"primary: {primary.error}"],
                optimized=False,
            )
            await self._record_result(payload, result, request_kind="fusion")
            return result

        primary_visible_content = strip_reasoning_text(primary.content)
        if not primary_visible_content and not primary.tool_calls:
            for task in aux_tasks:
                task.cancel()
            if aux_tasks:
                await asyncio.gather(*aux_tasks, return_exceptions=True)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result = FusionResult(
                message=self._primary_passthrough_message(primary, payload),
                debug={
                    "mode": "primary_passthrough_empty",
                    "skipped_judge": True,
                    "optimized": False,
                },
                primary=primary,
                aux=[],
                judge=None,
                usage=self._usage(primary, []),
                elapsed_ms=elapsed_ms,
                degraded=False,
                optimized=False,
            )
            await self._record_result(payload, result, request_kind="fusion")
            return result

        if not primary_visible_content and primary.tool_calls:
            for task in aux_tasks:
                task.cancel()
            if aux_tasks:
                await asyncio.gather(*aux_tasks, return_exceptions=True)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result = FusionResult(
                message=self._primary_passthrough_message(primary, payload),
                debug={
                    "mode": "primary_passthrough_tool_only",
                    "skipped_judge": True,
                    "optimized": False,
                },
                primary=primary,
                aux=[],
                judge=None,
                usage=self._usage(primary, []),
                elapsed_ms=elapsed_ms,
                degraded=False,
                optimized=False,
            )
            await self._record_result(payload, result, request_kind="fusion")
            return result

        aux = await asyncio.gather(*aux_tasks) if aux_tasks else []
        usable_aux: list[ModelResult] = []
        for result in aux:
            if result.error:
                errors.append(f"{result.display_name}: {result.error}")
            else:
                usable_aux.append(result)

        if not usable_aux:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result = FusionResult(
                message=self._primary_passthrough_message(primary, payload),
                debug={
                    "mode": "primary_passthrough_no_usable_aux",
                    "skipped_judge": True,
                    "aux_errors": errors,
                    "optimized": False,
                },
                primary=primary,
                aux=[],
                judge=None,
                usage=self._usage(primary, []),
                elapsed_ms=elapsed_ms,
                degraded=True,
                errors=errors or None,
            )
            await self._record_result(payload, result, request_kind="fusion")
            return result

        tool_names = available_tool_names(payload)
        judge_payload = {
            "messages": build_judge_messages(payload, primary, usable_aux, tool_names),
        }
        if payload.get("tools"):
            judge_payload["tools"] = payload["tools"]
            judge_payload["tool_choice"] = "auto"
        if "parallel_tool_calls" in payload:
            judge_payload["parallel_tool_calls"] = payload["parallel_tool_calls"]

        judge = await self._safe_model_call(
            self.config.models.judge,
            judge_payload,
            timeout_seconds=self.config.fusion.judge_timeout_seconds,
        )
        content_decision = parse_text_decision(judge.content)
        normalized_judge_tool_calls = None
        invalid_judge_tool_calls = False
        if judge.tool_calls:
            if not tool_names:
                invalid_judge_tool_calls = True
            else:
                normalized_judge_tool_calls = normalize_tool_calls(
                    judge.tool_calls,
                    allowed_names=tool_names,
                )
                invalid_judge_tool_calls = normalized_judge_tool_calls is None

        if judge.error or not content_decision or invalid_judge_tool_calls:
            if judge.error:
                errors.append(f"judge: {judge.error}")
            elif invalid_judge_tool_calls:
                errors.append("judge: returned invalid native tool_calls")
            else:
                errors.append("judge: returned invalid text decision")
            message = self._primary_passthrough_message(primary, payload)
            degraded = True
            optimized = False
            judge_decision = None
        else:
            judge_decision = {
                "content": content_decision,
                "tool_calls": {
                    "operation": "replace" if normalized_judge_tool_calls else "none",
                    "items": normalized_judge_tool_calls,
                },
            }
            primary_message = normalize_trajectory(
                primary.trajectory(),
                allowed_tool_names=tool_names,
                primary_trajectory=primary.trajectory(),
            )
            message = apply_hybrid_judge_update(
                primary.trajectory(),
                content_decision,
                allowed_tool_names=tool_names,
                judge_tool_calls=normalized_judge_tool_calls,
            )
            optimized = message != primary_message
            degraded = bool(errors)

        debug = {
            "mode": "judge_integrated_replacement",
            "judge_decision": judge_decision,
            "optimized": optimized,
        }
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        result = FusionResult(
            message=message,
            debug=debug,
            primary=primary,
            aux=usable_aux,
            judge=None if judge.error else judge,
            usage=self._usage(primary, usable_aux, None if judge.error else judge),
            elapsed_ms=elapsed_ms,
            degraded=degraded,
            errors=errors or None,
            optimized=optimized,
            judge_decision=judge_decision,
        )
        self._dump_debug(payload, result)
        await self._record_result(payload, result, request_kind="fusion")
        return result

    async def primary_only(self, payload: dict[str, Any]) -> FusionResult:
        start = time.perf_counter()
        primary = await self._safe_model_call(
            self.config.models.primary,
            payload,
            timeout_seconds=self.config.fusion.panel_timeout_seconds,
        )
        if primary.error:
            result = FusionResult(
                message=self._primary_passthrough_message(primary, payload),
                debug={
                    "mode": "primary_only_error_passthrough",
                    "primary_error": primary.error,
                },
                primary=primary,
                aux=[],
                judge=None,
                usage=self._usage(primary, []),
                elapsed_ms=int((time.perf_counter() - start) * 1000),
                degraded=True,
                errors=[f"primary: {primary.error}"],
                optimized=False,
            )
            self._dump_debug(payload, result)
            await self._record_result(payload, result, request_kind="primary_only")
            return result
        result = FusionResult(
            message=self._primary_passthrough_message(primary, payload),
            debug={"mode": "primary_only"},
            primary=primary,
            aux=[],
            judge=None,
            usage=self._usage(primary, []),
            elapsed_ms=int((time.perf_counter() - start) * 1000),
            optimized=False,
            judge_decision=None,
        )
        self._dump_debug(payload, result)
        await self._record_result(payload, result, request_kind="primary_only")
        return result

    async def _safe_model_call(
        self,
        model_config: BaseModelConfig,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> ModelResult:
        async with self._semaphore:
            client = OpenAICompatibleClient(
                model_config,
                timeout_seconds=timeout_seconds,
            )
            try:
                return await client.chat_completion(payload)
            except Exception as exc:
                return ModelResult(
                    role="assistant",
                    content=None,
                    tool_calls=None,
                    raw_message={},
                    raw_response={},
                    model=model_config.model_name,
                    url=model_config.url,
                    display_name=model_config.display_name,
                    latency_ms=0,
                    usage=None,
                    error=str(exc),
                )

    def _primary_passthrough_message(
        self,
        primary: ModelResult,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return normalize_trajectory(
            primary.trajectory(),
            allowed_tool_names=available_tool_names(payload),
            primary_trajectory=primary.trajectory(),
        )

    def _usage(
        self,
        primary: ModelResult,
        aux: list[ModelResult],
        judge: ModelResult | None = None,
    ) -> dict[str, Any]:
        calls = [primary, *aux]
        if judge is not None:
            calls.append(judge)

        aggregate = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        details = []
        for call in calls:
            usage = call.usage or {}
            details.append(
                {
                    "name": call.display_name,
                    "url": call.url,
                    "model": call.model,
                    "usage": usage,
                    "latency_ms": call.latency_ms,
                }
            )
            for key in aggregate:
                value = usage.get(key)
                if isinstance(value, int):
                    aggregate[key] += value

        return {
            **aggregate,
            "fusion_calls": details,
        }

    def _dump_debug(self, payload: dict[str, Any], result: FusionResult) -> None:
        if not self.config.fusion.debug_dump_dir:
            return
        dump_dir = Path(self.config.fusion.debug_dump_dir).expanduser()
        if not dump_dir.is_absolute():
            dump_dir = Path.cwd() / dump_dir
        dump_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = dump_dir / f"{timestamp}-{uuid.uuid4().hex[:8]}.json"
        data = {
            "elapsed_ms": result.elapsed_ms,
            "degraded": result.degraded,
            "errors": result.errors or [],
            "request_summary": {
                "model": payload.get("model"),
                "message_count": len(payload.get("messages") or []),
                "tool_names": available_tool_names(payload),
                "stream": payload.get("stream"),
            },
            "primary": {
                "model": result.primary.model,
                "display_name": result.primary.display_name,
                "trajectory": sanitize_trajectory(result.primary.trajectory()),
            },
            "aux": [
                {
                    "model": item.model,
                    "display_name": item.display_name,
                    "trajectory": sanitize_trajectory(item.trajectory()),
                }
                for item in result.aux
            ],
            "judge": {
                "model": result.judge.model if result.judge else None,
                "display_name": result.judge.display_name if result.judge else None,
                "decision": result.judge_decision,
            },
            "enhanced": sanitize_trajectory(result.message),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    async def record_stats(self) -> dict[str, Any]:
        record_dir = self._record_dir()
        if record_dir is None:
            stats = self._empty_record_stats()
            stats["enabled"] = False
            return stats
        async with self._record_lock:
            stats = self._read_record_stats(record_dir / "stats.json")
            stats["enabled"] = True
            stats["record_dir"] = str(record_dir)
            return stats

    async def _record_result(
        self,
        payload: dict[str, Any],
        result: FusionResult,
        *,
        request_kind: str,
    ) -> None:
        record_dir = self._record_dir()
        if record_dir is None:
            return

        async with self._record_lock:
            record_dir.mkdir(parents=True, exist_ok=True)
            stats_path = record_dir / "stats.json"
            stats = self._read_record_stats(stats_path)
            stats["enabled"] = True
            stats["record_dir"] = str(record_dir)
            stats["total_requests"] += 1
            if request_kind == "fusion":
                stats["fusion_requests"] += 1
                if result.optimized:
                    stats["fusion_optimized_requests"] += 1
                    stats["last_optimized_at"] = self._timestamp_iso()
                else:
                    stats["fusion_no_change_requests"] += 1
            else:
                stats["primary_only_requests"] += 1
            if result.degraded:
                stats["degraded_requests"] += 1
            stats["last_updated_at"] = self._timestamp_iso()
            self._refresh_record_rates(stats)

            if result.optimized:
                self._write_optimized_record(record_dir, payload, result, stats)
            self._write_json(stats_path, stats)

    def _write_optimized_record(
        self,
        record_dir: Path,
        payload: dict[str, Any],
        result: FusionResult,
        stats: dict[str, Any],
    ) -> None:
        optimized_dir = record_dir / "optimized"
        optimized_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = optimized_dir / f"{timestamp}-{uuid.uuid4().hex[:8]}.json"
        data = {
            "timestamp": self._timestamp_iso(),
            "elapsed_ms": result.elapsed_ms,
            "degraded": result.degraded,
            "errors": result.errors or [],
            "stats_after_request": {
                "total_requests": stats["total_requests"],
                "fusion_requests": stats["fusion_requests"],
                "fusion_optimized_requests": stats["fusion_optimized_requests"],
                "optimization_rate_all": stats["optimization_rate_all"],
                "optimization_rate_fusion": stats["optimization_rate_fusion"],
            },
            "request_summary": {
                "model": payload.get("model"),
                "message_count": len(payload.get("messages") or []),
                "tool_names": available_tool_names(payload),
                "stream": payload.get("stream"),
            },
            "primary": {
                "model": result.primary.model,
                "display_name": result.primary.display_name,
                "trajectory": sanitize_trajectory(result.primary.trajectory()),
            },
            "aux": [
                {
                    "model": item.model,
                    "display_name": item.display_name,
                    "trajectory": sanitize_trajectory(item.trajectory()),
                }
                for item in result.aux
            ],
            "judge": {
                "model": result.judge.model if result.judge else None,
                "display_name": result.judge.display_name if result.judge else None,
                "decision": result.judge_decision,
            },
            "enhanced": sanitize_trajectory(result.message),
        }
        self._write_json(path, data)

    def _record_dir(self) -> Path | None:
        if not self.config.fusion.record_dir:
            return None
        record_dir = Path(self.config.fusion.record_dir).expanduser()
        if not record_dir.is_absolute():
            record_dir = Path.cwd() / record_dir
        return record_dir

    def _read_record_stats(self, path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    stats = {**self._empty_record_stats(), **data}
                    self._refresh_record_rates(stats)
                    return stats
            except (OSError, json.JSONDecodeError):
                pass
        return self._empty_record_stats()

    def _empty_record_stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "record_dir": None,
            "total_requests": 0,
            "fusion_requests": 0,
            "primary_only_requests": 0,
            "fusion_optimized_requests": 0,
            "fusion_no_change_requests": 0,
            "degraded_requests": 0,
            "optimization_rate_all": 0.0,
            "optimization_rate_fusion": 0.0,
            "optimization_rate_all_percent": 0.0,
            "optimization_rate_fusion_percent": 0.0,
            "last_updated_at": None,
            "last_optimized_at": None,
        }

    def _refresh_record_rates(self, stats: dict[str, Any]) -> None:
        total = stats.get("total_requests") or 0
        fusion = stats.get("fusion_requests") or 0
        optimized = stats.get("fusion_optimized_requests") or 0
        all_rate = optimized / total if total else 0.0
        fusion_rate = optimized / fusion if fusion else 0.0
        stats["optimization_rate_all"] = round(all_rate, 6)
        stats["optimization_rate_fusion"] = round(fusion_rate, 6)
        stats["optimization_rate_all_percent"] = round(all_rate * 100, 2)
        stats["optimization_rate_fusion_percent"] = round(fusion_rate * 100, 2)

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp_path.replace(path)

    def _timestamp_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def completion_id() -> str:
    return "chatcmpl-fusion-" + uuid.uuid4().hex
