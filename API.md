# API

The local API is served by the FastAPI backend on `http://127.0.0.1:8000`.

This document covers the machine-to-machine routes exposed by the lab so another local program can:

- inspect health and monitoring state
- list, load, and unload Gemma runtimes
- send synchronous or streaming inference requests
- submit queued async requests and poll them later
- inspect request history persisted in SQLite

All examples below were revalidated locally on **April 5, 2026** against `Gemma 4 E2B / SFP8`.

## Base URL

```text
http://127.0.0.1:8000
```

No authentication is required on localhost.

## Runtime model selectors

The API accepts either:

- `model: "e2b:sfp8"` style selectors
- or separate `model_key` / `quantization_key`

Common selectors:

- `e2b:sfp8`
- `e2b:q4_0`
- `e4b:sfp8`
- `26b-a4b:sfp8`
- `26b-a4b:q4_0`
- `31b:sfp8`
- `31b:q4_0`
- `31b-nvfp4:nvfp4`

Important runtime note:

- the UI label `SFP8` is mapped in this repo to a practical local `Q8_0 GGUF` runtime through `llama.cpp`
- `NVFP4` is served through `WSL + vLLM`
- BF16 support depends on the local machine and is heavier than the quantized paths

## Health and control routes

### `GET /api/v1/health`

Returns the current service health, active model, load state, GPU identity, and basic runtime info.

```bash
curl http://127.0.0.1:8000/api/v1/health
```

### `GET /api/v1/status`

Returns a wider status object with:

- `health`
- `current_model`
- `queue`
- `capabilities`

### `GET /api/v1/capabilities`

Describes the exposed routes and high-level API features:

- control routes
- OpenAI-compatible routes
- multimodal support
- structured output support
- tool calling support
- async queue support

### `GET /api/v1/models`

Returns:

- `current_model`
- `load_state`
- `models`
- `quantizations`

### `GET /api/v1/models/current`

Returns only the currently loaded model state.

Example response shape:

```json
{
  "loaded": true,
  "current_model": {
    "model_key": "e2b",
    "quantization_key": "sfp8",
    "runtime_family": "llama.cpp",
    "runtime_capabilities": {
      "runtime_family": "llama.cpp",
      "supported_modalities": ["text", "image"],
      "supports_text": true,
      "supports_image": true,
      "supports_audio": false
    }
  },
  "load_state": {
    "status": "loaded",
    "progress": 100
  }
}
```

### `GET /api/v1/models/load-status`

Returns the model load progress only.

Useful for polling while another client triggers a model switch.

### `POST /api/v1/models/load`

Queues a model load.

Request:

```json
{
  "model_key": "e2b",
  "quantization_key": "sfp8"
}
```

Response:

```json
{
  "accepted": true,
  "message": "Queued Gemma 4 E2B in SFP8.",
  "load_state": {
    "status": "queued",
    "progress": 2
  },
  "current_model": null,
  "health": {
    "status": "idle"
  }
}
```

### `POST /api/v1/models/unload`

Unloads the currently active runtime.

Returns whether anything was unloaded plus the previous selection.

## Monitoring and request history

### `GET /api/v1/monitoring`

Returns a machine snapshot suitable for dashboards.

It includes:

- `health`
- `current_model`
- `gpu`
- `memory`
- `queue`
- `recent_requests`
- the SQLite request store path

Example `gpu` section:

```json
{
  "gpu": {
    "name": "NVIDIA GeForce RTX 5090",
    "utilization_gpu_percent": 15.0,
    "utilization_memory_percent": 1.0,
    "memory_used_gib": 9.65,
    "memory_total_gib": 31.84,
    "temperature_c": 32.0,
    "power_draw_watts": 72.86,
    "source": "nvidia-smi"
  }
}
```

### `GET /api/v1/requests`

Returns the recent request list from SQLite plus the current queue snapshot.

Optional query parameter:

- `limit` default `40`, max `200`

### `GET /api/v1/requests/{request_id}`

Returns one persisted request row and the current queue state.

Statuses used by the request store:

- `queued`
- `running`
- `completed`
- `failed`

## OpenAI-compatible routes

### `GET /v1/models`

Lists all exposed model and quantization combinations in an OpenAI-like format.

### `POST /api/v1/chat/completions`
### `POST /v1/chat/completions`

These two routes share the same backend behavior. The `/v1/...` path exists for compatibility.

Supported request fields:

```json
{
  "request_id": "optional-client-generated-id",
  "model": "e2b:sfp8",
  "messages": [
    { "role": "system", "content": "You are concise." },
    { "role": "user", "content": "Say hello." }
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 64,
  "stream": false,
  "thinking": false,
  "tts_enabled": false,
  "response_format": { "type": "text" },
  "tools": [],
  "tool_choice": "auto"
}
```

### Simple text example

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"e2b:sfp8\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: api text ok\"}],\"max_tokens\":24,\"temperature\":0}"
```

### Streaming example

Set `stream: true` to receive `text/event-stream`.

Each SSE event contains an OpenAI-style chunk. The stream ends with:

```text
data: [DONE]
```

The first chunk declares the assistant role, later chunks stream text in `choices[0].delta.content`.

### Structured JSON output

Use:

```json
{
  "response_format": { "type": "json_object" }
}
```

The gateway adds a JSON-output instruction and then validates the result.

Important caveat:

- this is **best effort JSON mode**
- it is not decoder-level constrained JSON
- the response still returns normal chat text fields plus `structured_output`

Example:

```json
{
  "model": "e2b:sfp8",
  "messages": [
    {
      "role": "user",
      "content": "Return a JSON object with keys ok and source. ok must be true."
    }
  ],
  "response_format": { "type": "json_object" }
}
```

### Tool calling

Use OpenAI-style function tool definitions:

```json
{
  "model": "e2b:sfp8",
  "messages": [
    {
      "role": "user",
      "content": "Use the tool get_time for Paris. Do not answer normally."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_time",
        "description": "Return the current time in a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": { "type": "string" }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

The response may include:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_...",
            "type": "function",
            "function": {
              "name": "get_time",
              "arguments": "{\"city\":\"Paris\"}"
            },
            "parsed_arguments": {
              "city": "Paris"
            }
          }
        ]
      }
    }
  ]
}
```

Important caveat:

- the model proposes the call
- your client executes the function
- your client then sends the tool result back in later messages if you want a second assistant pass

## Multimodal messages

`messages[].content` can be either a string or a list of typed parts.

Supported part types:

- `text`
- `image_url`
- `audio_url`

Example:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Describe this image." },
    { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }
  ]
}
```

Accepted media URL sources:

- `data:` URLs
- `file://` URLs
- absolute local file paths
- `http://` and `https://` URLs

Runtime caveat:

- model-level modality support and runtime-level modality support are not always identical
- for example, image works on the validated `llama.cpp` quantized path
- audio support depends on the active runtime and selected model
- the authoritative fields for the loaded runtime are `runtime_capabilities.supported_modalities`, `runtime_capabilities.supports_image`, and `runtime_capabilities.supports_audio`

Example:

- `e2b:sfp8` exposes `text + image` in the current `llama.cpp` runtime path
- the same Gemma family advertises audio at the model-card level, but this local quantized runtime currently rejects audio with HTTP `400`

## Async queue endpoint

### `POST /api/v1/requests/chat/completions`

This endpoint accepts the same request body as chat completions except `stream` must stay `false`.

It returns immediately with `202 Accepted`.

Example response:

```json
{
  "accepted": true,
  "request_id": "7aa1...",
  "status_url": "/api/v1/requests/7aa1...",
  "status": "queued",
  "queue": {
    "active_request_id": null,
    "queued_count": 1
  }
}
```

Then poll:

```bash
curl http://127.0.0.1:8000/api/v1/requests/7aa1...
```

The final row includes:

- request metadata
- timestamps
- request payload summary
- response payload or error

## Legacy UI routes

The dashboard still uses the older form-style routes, which remain exposed:

- `POST /api/generate`
- `POST /api/generate-stream`

They are useful if you want to emulate the web dashboard exactly, but new external clients should prefer `/api/v1/chat/completions` and `/api/v1/requests/chat/completions`.

## Persistence and logs

Request history is stored in:

```text
data/gemma4-requests.sqlite3
```

Operational logs are written under:

```text
logs/
```

Useful log files:

- `logs/gemma4-lab.log`
- `logs/llama-server.log`
- `logs/vllm-wsl.log`

## Smoke test

A reusable smoke test script is included:

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_test_api.py
```

Optional arguments:

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_test_api.py --base-url http://127.0.0.1:8000 --model e2b:sfp8
```

The script validates:

- automatic load of the requested model if needed
- health and status routes
- model listing and current selection
- monitoring snapshot
- synchronous chat
- JSON mode
- tool calls
- image input
- SSE streaming
- async queue submission and polling
- legacy dashboard routes

## What was verified on April 5, 2026

The following passed locally on this workstation:

- `GET /api/v1/health`
- `GET /api/v1/status`
- `GET /api/v1/capabilities`
- `GET /api/v1/models`
- `GET /api/v1/models/current`
- `GET /api/v1/models/load-status`
- `GET /api/v1/monitoring`
- `GET /v1/models`
- `POST /api/v1/chat/completions`
- `POST /v1/chat/completions` with SSE stream
- `POST /api/v1/requests/chat/completions`
- `GET /api/v1/requests/{request_id}`
- `POST /api/generate`
- `POST /api/generate-stream`

The same session also revalidated:

- unload and reload through `/api/v1/models/unload` and `/api/v1/models/load`
- persisted request history in SQLite
- queue status reporting while a request is in progress
