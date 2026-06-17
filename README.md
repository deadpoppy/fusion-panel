# Fusion Panel

Fusion Panel is an OpenAI-compatible chat completions proxy that runs one
primary model, several auxiliary models, and a judge model as a small model
panel. The primary model remains the anchor; auxiliary trajectories are used to
catch missing context, contradictions, partial coverage, and useful side
signals. When the primary answer is already good enough, Fusion Panel returns it
unchanged.

The server exposes:

```text
POST http://127.0.0.1:8082/v1/chat/completions
GET  http://127.0.0.1:8082/health
GET  http://127.0.0.1:8082/fusion/stats
```

## Quick Start

Clone the repo, create one config file, then start:

```bash
cp config.yaml.example config.yaml
vim config.yaml
./start-fusion-panel.sh
```

The start script creates `.venv`, installs the local package, and runs the
server. You can override host or port after the script:

```bash
./start-fusion-panel.sh --host 0.0.0.0 --port 8082
```

If you prefer manual installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
fusion-panel --config config.yaml
```

The legacy command name is also available:

```bash
trajectory-fusion --config config.yaml
```

## Configure

Fusion Panel needs one YAML config. Values support environment expansion with
`${NAME}` and `${NAME:-default}`:

```yaml
server:
  host: 127.0.0.1
  port: 8082
  request_timeout_seconds: 480

fusion:
  enabled_by_default: true
  delayed_streaming: true
  panel_timeout_seconds: 480
  judge_timeout_seconds: 480
  debug_dump_dir: fusion-dumps
  record_dir: fusion-records

models:
  primary:
    url: https://open.bigmodel.cn/api/coding/paas/v4
    api_key: ""
    model_name: glm-5.2
  aux:
    - name: aux-mini-a
      url: https://api.minimaxi.com/v1
      api_key: ""
      model_name: MiniMax-M2.7
      temperature: 0.7
    - name: aux-mini-b
      url: https://api.minimaxi.com/v1
      api_key: ""
      model_name: MiniMax-M3
      temperature: 0.8
  judge:
    url: https://api.minimaxi.com/v1
    api_key: ""
    model_name: MiniMax-M3
    temperature: 0.3
```

Each model entry has its own `url`, `api_key`, `model_name`, optional `name`,
optional `organization`, optional `extra_headers`, and optional `extra_body`.
`url` can be either a base URL such as `https://api.openai.com/v1` or a full
`.../chat/completions` endpoint.

`config.yaml` is ignored by git so local keys stay local.

## Request

```bash
curl http://127.0.0.1:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @examples/request.json
```

Streaming uses delayed SSE: the connection opens immediately, heartbeat frames
are sent while the panel is running, and the final fused answer is streamed once
ready.

```bash
curl -N http://127.0.0.1:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fusion-panel",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Draft a migration plan from SQLite to Postgres."}
    ]
  }'
```

Per-request options:

```json
{
  "fusion": false,
  "include_fusion_debug": true
}
```

## Tool Calls

Fusion Panel preserves OpenAI-compatible `tool_calls`.

For requests with tools:

1. The primary and auxiliary models each produce a full assistant trajectory.
2. The judge evaluates whether the primary has comprehensive coverage.
3. If the primary is incomplete, the judge replaces primary content with an
   integrated answer in the primary style.
4. If judge-native `tool_calls` are valid, they replace primary tool calls.
5. If judge tool calls are invalid, Fusion Panel falls back to the primary
   trajectory.

See `examples/tool_request.json` for a minimal tool-call request.

## Fusion Records

Set `fusion.record_dir` to enable lightweight long-running stats:

```yaml
fusion:
  record_dir: fusion-records
```

The server writes:

- `fusion-records/stats.json` for counters and optimization rates.
- `fusion-records/optimized/*.json` for requests where the judge replaced the
  primary trajectory.

Runtime records, debug dumps, logs, pid files, and local config are ignored by
git.

## Test

```bash
python tests/test_smoke.py
```

## Project Layout

```text
src/trajectory_fusion/   Python package
config.yaml.example      Copy this to config.yaml
examples/                Example OpenAI-compatible requests
start-fusion-panel.sh    One-command local launcher
tests/                   Smoke tests with mocked upstream models
```
