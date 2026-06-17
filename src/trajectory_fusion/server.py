from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import AppConfig, load_config
from .engine import FusionEngine
from .responses import (
    chat_completion_response,
    delayed_stream_response_with_heartbeat,
)


class AppState:
    config: AppConfig
    engine: FusionEngine


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(state, "config"):
        state.config = load_config()
        state.engine = FusionEngine(state.config)
    app.state.config = state.config
    app.state.engine = state.engine
    yield


app = FastAPI(title="Fusion Panel", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "fusion-panel"}


@app.get("/fusion/stats")
async def fusion_stats(request: Request) -> dict[str, Any]:
    engine: FusionEngine = request.app.state.engine
    return await engine.record_stats()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc

    config: AppConfig = request.app.state.config
    engine: FusionEngine = request.app.state.engine

    stream = bool(payload.get("stream", False))
    fusion_enabled = bool(payload.get("fusion", config.fusion.enabled_by_default))
    include_debug = bool(
        payload.get("include_fusion_debug", config.fusion.include_debug_in_response)
    )
    response_model = payload.get("model") or config.models.primary.model_name

    async def resolve_result():
        try:
            if fusion_enabled:
                return await engine.run(payload)
            return await engine.primary_only(payload)
        except Exception as exc:
            if not config.fusion.fallback_to_primary_on_failure or not fusion_enabled:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            try:
                result = await engine.primary_only(payload)
                result.degraded = True
                result.errors = [str(exc)]
                return result
            except Exception as primary_exc:
                raise HTTPException(status_code=502, detail=str(primary_exc)) from primary_exc

    if stream:
        return StreamingResponse(
            delayed_stream_response_with_heartbeat(
                resolve_result(),
                model=response_model,
                heartbeat_seconds=config.fusion.stream_heartbeat_seconds,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    result = await resolve_result()

    headers = {}
    if config.fusion.expose_debug_headers:
        headers = {
            "x-fusion-elapsed-ms": str(result.elapsed_ms),
            "x-fusion-degraded": "true" if result.degraded else "false",
            "x-fusion-optimized": "true" if result.optimized else "false",
        }

    return JSONResponse(
        chat_completion_response(
            result,
            model=response_model,
            include_debug=include_debug,
        ),
        headers=headers,
    )


def build_app(config: AppConfig) -> FastAPI:
    state.config = config
    state.engine = FusionEngine(config)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fusion Panel proxy.")
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    parser.add_argument("--host", default=None, help="Override configured host.")
    parser.add_argument("--port", type=int, default=None, help="Override configured port.")
    args = parser.parse_args()

    config = load_config(args.config)
    build_app(config)
    uvicorn.run(
        app,
        host=args.host or config.server.host,
        port=args.port or config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
