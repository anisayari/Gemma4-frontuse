from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import requests


PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS0cA"
    "AAAASUVORK5CYII="
)


def record(results: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    results.append({"name": name, "ok": ok, "detail": detail})
    state = "PASS" if ok else "FAIL"
    print(f"[{state}] {name}: {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the Gemma 4 local API.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the local API.",
    )
    parser.add_argument(
        "--model",
        default="e2b:sfp8",
        help="Model selector used for chat tests, for example e2b:sfp8.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds per request.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    timeout = args.timeout
    results: list[dict[str, Any]] = []
    session = requests.Session()
    data_url = f"data:image/png;base64,{PNG_B64}"

    def request(method: str, path: str, **kwargs: Any) -> requests.Response:
        return session.request(method, f"{base_url}{path}", timeout=timeout, **kwargs)

    def ensure_model_loaded() -> None:
        model_key, quantization_key = args.model.split(":", 1)
        current = request("GET", "/api/v1/models/current")
        current_payload = current.json()
        current_model = current_payload.get("current_model") or {}
        if (
            current_payload.get("loaded")
            and current_model.get("model_key") == model_key
            and current_model.get("quantization_key") == quantization_key
        ):
            record(results, "ensure_model_loaded", True, f"already loaded: {args.model}")
            return

        load = request(
            "POST",
            "/api/v1/models/load",
            json={"model_key": model_key, "quantization_key": quantization_key},
        )
        if not load.ok:
            detail = load.text[:200]
            record(results, "ensure_model_loaded", False, detail)
            raise RuntimeError(detail)

        deadline = time.time() + timeout
        final_state = None
        while time.time() < deadline:
            status = request("GET", "/api/v1/models/load-status")
            final_state = status.json().get("load_state", {})
            if final_state.get("status") in {"loaded", "failed", "idle"}:
                break
            time.sleep(1)

        ok = (
            isinstance(final_state, dict)
            and final_state.get("status") == "loaded"
            and final_state.get("target_model_key") == model_key
            and final_state.get("target_quantization_key") == quantization_key
        )
        detail = (
            f"status={final_state.get('status')} target="
            f"{final_state.get('target_model_key')}:{final_state.get('target_quantization_key')}"
        )
        record(results, "ensure_model_loaded", ok, detail)
        if not ok:
            raise RuntimeError(detail)

    try:
        ensure_model_loaded()

        health = request("GET", "/api/v1/health")
        payload = health.json()
        record(results, "health", health.ok and "status" in payload, f"status={payload.get('status')}")

        status = request("GET", "/api/v1/status")
        payload = status.json()
        record(
            results,
            "status",
            status.ok and "queue" in payload and "capabilities" in payload,
            f"loaded={payload.get('health', {}).get('loaded')}",
        )

        caps = request("GET", "/api/v1/capabilities")
        payload = caps.json()
        record(
            results,
            "capabilities",
            caps.ok and "openai_compatible_routes" in payload,
            f"chat={payload.get('openai_compatible_routes', {}).get('chat_completions')}",
        )

        models = request("GET", "/api/v1/models")
        payload = models.json()
        record(
            results,
            "models",
            models.ok and isinstance(payload.get("models"), list),
            f"models={len(payload.get('models', []))} quants={len(payload.get('quantizations', []))}",
        )

        current = request("GET", "/api/v1/models/current")
        payload = current.json()
        record(
            results,
            "current_model",
            current.ok and "loaded" in payload,
            f"loaded={payload.get('loaded')} current={payload.get('current_model', {}).get('model_key')}",
        )

        load_status = request("GET", "/api/v1/models/load-status")
        payload = load_status.json()
        record(
            results,
            "load_status",
            load_status.ok and "load_state" in payload,
            f"status={payload.get('load_state', {}).get('status')}",
        )

        monitoring = request("GET", "/api/v1/monitoring")
        payload = monitoring.json()
        record(
            results,
            "monitoring",
            monitoring.ok and "gpu" in payload and "memory" in payload,
            f"gpu_used={payload.get('gpu', {}).get('memory_used_gib')}",
        )

        openai_models = request("GET", "/v1/models")
        payload = openai_models.json()
        record(
            results,
            "openai_models",
            openai_models.ok and payload.get("object") == "list",
            f"count={len(payload.get('data', []))}",
        )

        sync_text = request(
            "POST",
            "/api/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": "Reply with exactly: api text ok"}],
                "max_tokens": 24,
                "temperature": 0,
            },
        )
        payload = sync_text.json()
        message = payload["choices"][0]["message"]["content"]
        record(
            results,
            "chat_text",
            sync_text.ok and "api text ok" in message.lower(),
            message[:120],
        )

        json_mode = request(
            "POST",
            "/api/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Return a JSON object with keys ok and source. ok must be true.",
                    }
                ],
                "max_tokens": 96,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        payload = json_mode.json()
        structured = payload.get("structured_output")
        record(
            results,
            "chat_json_mode",
            json_mode.ok and isinstance(structured, dict) and structured.get("ok") is True,
            json.dumps(structured, ensure_ascii=False)[:140],
        )

        tool_call = request(
            "POST",
            "/api/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [
                    {"role": "user", "content": "Use the tool get_time for Paris. Do not answer normally."}
                ],
                "max_tokens": 128,
                "temperature": 0,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "description": "Return the current time in a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        )
        payload = tool_call.json()
        tool_calls = payload["choices"][0]["message"].get("tool_calls")
        record(
            results,
            "chat_tools",
            tool_call.ok and isinstance(tool_calls, list) and len(tool_calls) > 0,
            json.dumps(tool_calls, ensure_ascii=False)[:180],
        )

        image_call = request(
            "POST",
            "/api/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in a few words."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                "max_tokens": 48,
                "temperature": 0,
            },
        )
        payload = image_call.json()
        message = payload["choices"][0]["message"]["content"]
        record(
            results,
            "chat_image",
            image_call.ok and isinstance(message, str) and bool(message.strip()),
            message[:120],
        )

        stream = session.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": "Count from one to five in French, just the words."}],
                "max_tokens": 48,
                "temperature": 0,
                "stream": True,
            },
            timeout=timeout,
            stream=True,
        )
        got_token = False
        finish_reason = None
        with stream as response:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                body = raw_line[6:]
                if body == "[DONE]":
                    break
                payload = json.loads(body)
                if not payload.get("choices"):
                    continue
                delta = payload["choices"][0].get("delta", {})
                if delta.get("content"):
                    got_token = True
                if payload["choices"][0].get("finish_reason"):
                    finish_reason = payload["choices"][0]["finish_reason"]
        record(results, "chat_stream", stream.ok and got_token, f"finish={finish_reason}")

        async_submit = request(
            "POST",
            "/api/v1/requests/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": "Reply with exactly: async ok"}],
                "max_tokens": 24,
                "temperature": 0,
            },
        )
        payload = async_submit.json()
        request_id = payload.get("request_id")
        async_ok = async_submit.status_code == 202 and bool(request_id)
        detail = f"request_id={request_id}"
        if async_ok:
            deadline = time.time() + timeout
            final_status = None
            while time.time() < deadline:
                request_status = request("GET", f"/api/v1/requests/{request_id}")
                request_payload = request_status.json()["request"]
                final_status = request_payload.get("status")
                if final_status in {"completed", "failed"}:
                    break
                time.sleep(1)
            async_ok = async_ok and final_status == "completed"
            detail = f"status={final_status}"
        record(results, "async_queue", async_ok, detail)

        request_list = request("GET", "/api/v1/requests?limit=5")
        payload = request_list.json()
        record(
            results,
            "requests_list",
            request_list.ok and isinstance(payload.get("requests"), list),
            f"returned={len(payload.get('requests', []))}",
        )

        legacy_generate = request(
            "POST",
            "/api/generate",
            data={
                "model_key": args.model.split(":")[0],
                "quantization_key": args.model.split(":")[1],
                "prompt": "Reply with exactly: legacy ok",
                "system_prompt": "You are concise.",
                "history_json": "[]",
                "max_new_tokens": "24",
                "temperature": "0",
            },
        )
        payload = legacy_generate.json()
        reply = payload.get("reply", "")
        record(
            results,
            "legacy_generate",
            legacy_generate.ok and "legacy ok" in reply.lower(),
            reply[:120],
        )

        legacy_stream = session.post(
            f"{base_url}/api/generate-stream",
            data={
                "model_key": args.model.split(":")[0],
                "quantization_key": args.model.split(":")[1],
                "prompt": "Reply with exactly: legacy stream ok",
                "system_prompt": "You are concise.",
                "history_json": "[]",
                "max_new_tokens": "32",
                "temperature": "0",
            },
            timeout=timeout,
            stream=True,
        )
        got_start = False
        got_token = False
        got_done = False
        with legacy_stream as response:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                payload = json.loads(raw_line)
                if payload.get("event") == "start":
                    got_start = True
                elif payload.get("event") == "token" and payload.get("text"):
                    got_token = True
                elif payload.get("event") == "done":
                    got_done = True
        record(
            results,
            "legacy_generate_stream",
            legacy_stream.ok and got_start and got_token and got_done,
            f"start={got_start} token={got_token} done={got_done}",
        )
    except Exception as exc:
        record(results, "smoke_test_runtime", False, repr(exc))

    failed = [item["name"] for item in results if not item["ok"]]
    print("\nSUMMARY")
    print(json.dumps({"total": len(results), "failed": failed}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
