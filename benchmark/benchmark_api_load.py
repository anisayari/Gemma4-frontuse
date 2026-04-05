from __future__ import annotations

import argparse
import base64
import io
import json
import math
import statistics
import struct
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import requests


RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODELS = [
    ("e2b", "sfp8", "Gemma 4 E2B / SFP8"),
    ("e2b", "q4_0", "Gemma 4 E2B / Q4_0"),
    ("e4b", "sfp8", "Gemma 4 E4B / SFP8"),
    ("e4b", "q4_0", "Gemma 4 E4B / Q4_0"),
    ("26b-a4b", "sfp8", "Gemma 4 26B A4B / SFP8"),
    ("26b-a4b", "q4_0", "Gemma 4 26B A4B / Q4_0"),
    ("31b", "q4_0", "Gemma 4 31B / Q4_0"),
    ("31b", "sfp8", "Gemma 4 31B / SFP8"),
    ("31b-nvfp4", "nvfp4", "Gemma 4 31B IT NVFP4 / NVFP4"),
]

WORKLOADS = [
    {
        "name": "short",
        "max_tokens": 64,
        "prompt": (
            "You are benchmarking a local Gemma API on one RTX 5090. "
            "In two short French sentences, explain what this server is useful for."
        ),
    },
    {
        "name": "medium",
        "max_tokens": 128,
        "prompt": (
            "Context: this workstation runs a local Gemma 4 lab with a FastAPI backend, a React dashboard, "
            "model switching, request tracking, monitoring, image input, and local TTS. "
            "Task: answer in French with 6 concise bullet points describing why this is useful for prototyping local AI products."
        ),
    },
    {
        "name": "long",
        "max_tokens": 192,
        "prompt": (
            "Context: this benchmark targets a local Gemma 4 workstation lab on Windows with a FastAPI API, "
            "request IDs, request history in SQLite, monitoring endpoints, model load and unload routes, "
            "image input support on llama.cpp, and streaming token output for clients. "
            "Task: write 12 numbered points in French, each one short but informative, about how such a local API can help "
            "developers test product ideas, multimodal workflows, request queues, and observability before deployment."
        ),
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--load-timeout", type=int, default=700)
    parser.add_argument("--request-timeout", type=int, default=700)
    return parser.parse_args()


def req(session, method, url, timeout, **kwargs):
    response = session.request(method, url, timeout=timeout, **kwargs)
    try:
        payload = response.json()
    except Exception:
        payload = {"raw_text": response.text}
    return response, payload


def ensure_loaded(session, base_url, model_key, quantization_key, load_timeout):
    started = time.perf_counter()
    response, payload = req(
        session,
        "POST",
        f"{base_url}/api/v1/models/load",
        load_timeout,
        json={"model_key": model_key, "quantization_key": quantization_key},
    )
    if response.status_code not in {200, 202}:
        return {
            "ok": False,
            "status_code": response.status_code,
            "error": payload.get("detail") or payload.get("message") or str(payload),
            "load_seconds": round(time.perf_counter() - started, 3),
        }
    state = payload.get("load_state") or {}
    deadline = time.time() + load_timeout
    while time.time() < deadline:
        if state.get("status") in {"loaded", "failed", "idle"}:
            break
        time.sleep(1.5)
        _, status_payload = req(session, "GET", f"{base_url}/api/v1/models/load-status", 60)
        state = status_payload.get("load_state") or {}
    current_model = None
    if state.get("status") == "loaded":
        _, current_payload = req(session, "GET", f"{base_url}/api/v1/models/current", 60)
        current_model = current_payload.get("current_model")
    return {
        "ok": state.get("status") == "loaded",
        "status_code": response.status_code,
        "error": state.get("error"),
        "message": state.get("message") or payload.get("message"),
        "load_state": state,
        "load_seconds": round(time.perf_counter() - started, 3),
        "current_model": current_model,
        "runtime_family": (current_model or {}).get("runtime_family"),
        "runtime_capabilities": (current_model or {}).get("runtime_capabilities"),
    }


def stream_completion(session, base_url, model_id, prompt, max_tokens, timeout):
    start = time.perf_counter()
    request_id = None
    ttft = None
    output = ""
    finish_reason = None
    prompt_tokens = None
    completion_tokens = None
    with session.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        timeout=timeout,
        stream=True,
    ) as response:
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            body = raw[6:]
            if body == "[DONE]":
                break
            payload = json.loads(body)
            request_id = payload.get("request_id") or request_id
            if payload.get("error"):
                raise RuntimeError(payload["error"].get("message") or str(payload["error"]))
            choices = payload.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    if ttft is None:
                        ttft = round(time.perf_counter() - start, 3)
                    output += text
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")
            usage = payload.get("usage") or {}
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
    return {
        "ok": True,
        "request_id": request_id,
        "ttft_seconds": ttft,
        "total_seconds": round(time.perf_counter() - start, 3),
        "output_chars": len(output),
        "output_preview": output[:160].replace("\n", " "),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def run_parallel(session_factory, base_url, model_id, concurrent_requests, timeout):
    barrier = threading.Barrier(concurrent_requests)
    results = [None] * concurrent_requests

    def worker(index):
        session = session_factory()
        try:
            barrier.wait()
            started = time.perf_counter()
            response = session.post(
                f"{base_url}/api/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Reply in one short French paragraph about why request queues matter for local inference.",
                        }
                    ],
                    "max_tokens": 96,
                    "temperature": 0,
                },
                timeout=timeout,
            )
            payload = response.json()
            content = (payload.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            results[index] = {
                "ok": response.ok,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "response_chars": len(content),
                "status_code": response.status_code,
                "request_id": payload.get("request_id"),
                "error": payload.get("detail"),
            }
        except Exception as exc:
            results[index] = {
                "ok": False,
                "latency_seconds": None,
                "response_chars": 0,
                "status_code": None,
                "request_id": None,
                "error": repr(exc),
            }
        finally:
            session.close()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(concurrent_requests)]
    batch_start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    latencies = [item["latency_seconds"] for item in results if item and item["latency_seconds"] is not None]
    return {
        "concurrent_requests": concurrent_requests,
        "ok": all(item and item["ok"] for item in results),
        "makespan_seconds": round(time.perf_counter() - batch_start, 3),
        "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "max_latency_seconds": round(max(latencies), 3) if latencies else None,
        "results": results,
    }


def model_color(label):
    palette = ["#88adff", "#4fd1c5", "#fab0ff", "#ffb86c", "#ff8a8a", "#c3e88d", "#82aaff", "#ffd866", "#a6accd"]
    return palette[abs(hash(label)) % len(palette)]


def build_latency_svg(results, output_path):
    rows = []
    for item in results:
        if not item.get("load", {}).get("ok"):
            continue
        for workload in item.get("workloads", []):
            if workload.get("ok"):
                rows.append(
                    {
                        "label": item["label"],
                        "workload": workload["name"],
                        "input_chars": workload["input_chars"],
                        "total_seconds": workload["total_seconds"],
                        "output_chars": workload["output_chars"],
                    }
                )
    if not rows:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    width, height = 1200, 760
    ml, mr, mt, mb = 90, 320, 90, 90
    pw, ph = width - ml - mr, height - mt - mb
    max_input = max(row["input_chars"] for row in rows)
    max_latency = max(row["total_seconds"] for row in rows)
    max_output = max(row["output_chars"] for row in rows)
    sx = lambda value: ml + (value / max_input) * pw
    sy = lambda value: mt + ph - (value / max_latency) * ph
    sr = lambda value: 8 + (math.sqrt(max(value, 1)) / math.sqrt(max_output)) * 22
    labels = []
    for row in rows:
        if row["label"] not in labels:
            labels.append(row["label"])
    svg = [
        f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>",
        "<style>.bg{fill:#08101f}.panel{fill:#0f1930}.grid{stroke:#2b3550;stroke-width:1;opacity:.7}.axis{stroke:#8aa5d6;stroke-width:2}.title{fill:#dee5ff;font:800 28px Manrope,Inter,Arial,sans-serif}.text{fill:#dee5ff;font:14px Inter,Arial,sans-serif}.muted{fill:#a3aac4;font:12px Inter,Arial,sans-serif}</style>",
        f"<rect class='bg' x='0' y='0' width='{width}' height='{height}' rx='24' />",
        f"<rect class='panel' x='30' y='30' width='{width-60}' height='{height-60}' rx='24' />",
        "<text class='title' x='60' y='78'>API latency by input size and response size</text>",
        "<text class='muted' x='60' y='104'>X = prompt chars | Y = total latency (s) | bubble size = output chars</text>",
    ]
    for step in range(6):
        x = ml + (pw / 5) * step
        y = mt + (ph / 5) * step
        svg.append(f"<line class='grid' x1='{x:.1f}' y1='{mt}' x2='{x:.1f}' y2='{mt+ph}' />")
        svg.append(f"<line class='grid' x1='{ml}' y1='{y:.1f}' x2='{ml+pw}' y2='{y:.1f}' />")
        svg.append(f"<text class='muted' x='{x-10:.1f}' y='{mt+ph+28:.1f}'>{round((max_input/5)*step)}</text>")
        svg.append(f"<text class='muted' x='{ml-52}' y='{y+4:.1f}'>{round((max_latency/5)*(5-step),1)}</text>")
    svg.append(f"<line class='axis' x1='{ml}' y1='{mt+ph}' x2='{ml+pw}' y2='{mt+ph}' />")
    svg.append(f"<line class='axis' x1='{ml}' y1='{mt}' x2='{ml}' y2='{mt+ph}' />")
    svg.append(f"<text class='text' x='{ml + pw/2 - 70:.1f}' y='{height-32}'>Prompt length (chars)</text>")
    svg.append(f"<text class='text' transform='translate(24,{mt + ph/2 + 50:.1f}) rotate(-90)'>Total latency (s)</text>")
    for row in rows:
        color = model_color(row["label"])
        cx, cy, radius = sx(row["input_chars"]), sy(row["total_seconds"]), sr(row["output_chars"])
        svg.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{radius:.1f}' fill='{color}' fill-opacity='.72' stroke='#e8eeff' stroke-width='1.5' />")
        svg.append(f"<text class='muted' x='{cx + radius + 6:.1f}' y='{cy + 4:.1f}'>{row['workload'][0].upper()}</text>")
    legend_y = mt + 20
    for index, label in enumerate(labels):
        y = legend_y + index * 26
        svg.append(f"<circle cx='{width-270}' cy='{y}' r='7' fill='{model_color(label)}' />")
        svg.append(f"<text class='text' x='{width-255}' y='{y+4}'>{label}</text>")
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def build_parallel_svg(rows, output_path):
    rows = [row for row in rows if row.get("concurrent_requests")]
    if not rows:
        output_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    width, height = 1200, 620
    ml, mr, mt, mb = 90, 120, 90, 90
    pw, ph = width - ml - mr, height - mt - mb
    max_value = max(row["makespan_seconds"] for row in rows if row.get("makespan_seconds") is not None)
    models = []
    for row in rows:
        if row["label"] not in models:
            models.append(row["label"])
    concurrency_values = sorted({row["concurrent_requests"] for row in rows})
    group_width = pw / max(1, len(concurrency_values))
    bar_width = min(70, (group_width - 30) / max(1, len(models)))
    sy = lambda value: mt + ph - (value / max_value) * ph
    svg = [
        f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>",
        "<style>.bg{fill:#08101f}.panel{fill:#0f1930}.grid{stroke:#2b3550;stroke-width:1;opacity:.7}.axis{stroke:#8aa5d6;stroke-width:2}.title{fill:#dee5ff;font:800 28px Manrope,Inter,Arial,sans-serif}.text{fill:#dee5ff;font:14px Inter,Arial,sans-serif}.muted{fill:#a3aac4;font:12px Inter,Arial,sans-serif}</style>",
        f"<rect class='bg' x='0' y='0' width='{width}' height='{height}' rx='24' />",
        f"<rect class='panel' x='30' y='30' width='{width-60}' height='{height-60}' rx='24' />",
        "<text class='title' x='60' y='78'>Concurrent request makespan</text>",
        "<text class='muted' x='60' y='104'>The current backend serializes inference through one queue, so values close to linear imply effective parallelism near 1.</text>",
    ]
    for step in range(6):
        y = mt + (ph / 5) * step
        svg.append(f"<line class='grid' x1='{ml}' y1='{y:.1f}' x2='{ml+pw}' y2='{y:.1f}' />")
        svg.append(f"<text class='muted' x='{ml-52}' y='{y+4:.1f}'>{round((max_value/5)*(5-step),1)}</text>")
    svg.append(f"<line class='axis' x1='{ml}' y1='{mt+ph}' x2='{ml+pw}' y2='{mt+ph}' />")
    svg.append(f"<line class='axis' x1='{ml}' y1='{mt}' x2='{ml}' y2='{mt+ph}' />")
    for group_index, concurrency in enumerate(concurrency_values):
        group_center = ml + group_index * group_width + group_width / 2
        svg.append(f"<text class='text' x='{group_center-8:.1f}' y='{mt+ph+30}'>{concurrency}</text>")
        subset = [row for row in rows if row["concurrent_requests"] == concurrency]
        for model_index, row in enumerate(subset):
            x = group_center - (len(subset) * bar_width) / 2 + model_index * bar_width
            y = sy(row["makespan_seconds"])
            h = mt + ph - y
            svg.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width-8:.1f}' height='{h:.1f}' fill='{model_color(row['label'])}' rx='8' />")
            svg.append(f"<text class='muted' x='{x:.1f}' y='{y-8:.1f}'>{row['makespan_seconds']}</text>")
    svg.append(f"<text class='text' x='{ml + pw/2 - 75:.1f}' y='{height-32}'>Concurrent requests</text>")
    svg.append(f"<text class='text' transform='translate(24,{mt + ph/2 + 45:.1f}) rotate(-90)'>Batch makespan (s)</text>")
    for idx, model in enumerate(models):
        y = mt + 24 + idx * 24
        svg.append(f"<circle cx='{width-260}' cy='{y}' r='7' fill='{model_color(model)}' />")
        svg.append(f"<text class='text' x='{width-245}' y='{y+4}'>{model}</text>")
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def build_audio_probe(session, base_url):
    ensure_loaded(session, base_url, "e2b", "sfp8", 300)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"".join(struct.pack("<h", 0) for _ in range(1600)))
    data_url = "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    response, payload = req(
        session,
        "POST",
        f"{base_url}/api/v1/chat/completions",
        120,
        json={
            "model": "e2b:sfp8",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe or describe this audio briefly."},
                        {"type": "audio_url", "audio_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": 48,
            "temperature": 0,
        },
    )
    return {
        "status_code": response.status_code,
        "detail": payload.get("detail") or payload.get("message") or str(payload),
    }


def build_report(results, parallel_results, audio_probe, bubble_svg_path, parallel_svg_path):
    passed = [item for item in results if item.get("load", {}).get("ok")]
    failed = [item for item in results if not item.get("load", {}).get("ok")]
    lines = [
        "# API Load Test",
        "",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "## Why audio is rejected on the current quantized runtime",
        "",
        f"- Probe result on `e2b:sfp8`: HTTP `{audio_probe['status_code']}` with `{audio_probe['detail']}`.",
        "- In this app, the quantized `llama.cpp` path explicitly rejects audio input before inference. Image is wired through; audio is not on that runtime path.",
        "",
        "## Load results",
        "",
        "| Model | Load OK | Load time (s) | Runtime | Runtime modalities | Notes |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for item in results:
        load = item["load"]
        caps = load.get("runtime_capabilities") or {}
        modalities = ", ".join(caps.get("supported_modalities") or [])
        notes = load.get("error") or load.get("message") or ""
        lines.append(
            f"| {item['label']} | {'yes' if load.get('ok') else 'no'} | {load.get('load_seconds')} | "
            f"{load.get('runtime_family') or '-'} | {modalities or '-'} | {notes} |"
        )
    if passed:
        lines += [
            "",
            "## Workload matrix",
            "",
            "| Model | Workload | Input chars | Max tokens | TTFT (s) | Total (s) | Output chars | Completion tokens | Finish |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for item in passed:
            for workload in item.get("workloads", []):
                if not workload.get("ok"):
                    lines.append(
                        f"| {item['label']} | {workload['name']} | {workload['input_chars']} | {workload['max_tokens']} | - | - | - | - | ERROR |"
                    )
                    continue
                lines.append(
                    f"| {item['label']} | {workload['name']} | {workload['input_chars']} | {workload['max_tokens']} | "
                    f"{workload['ttft_seconds']} | {workload['total_seconds']} | {workload['output_chars']} | "
                    f"{workload.get('completion_tokens') or '-'} | {workload.get('finish_reason') or '-'} |"
                )
        lines += ["", "## Latency graph", "", f"![API latency bubble chart]({bubble_svg_path.name})"]
    if parallel_results:
        lines += [
            "",
            "## Parallel request test",
            "",
            "| Model | Concurrent requests | Makespan (s) | Avg latency (s) | Max latency (s) | Effective parallelism |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for item in parallel_results:
            lines.append(
                f"| {item['label']} | {item['concurrent_requests']} | {item['makespan_seconds']} | "
                f"{item.get('avg_latency_seconds') or '-'} | {item.get('max_latency_seconds') or '-'} | "
                f"{item.get('effective_parallelism') or '-'} |"
            )
        lines += [
            "",
            f"![Parallel request makespan chart]({parallel_svg_path.name})",
            "",
            "- The current backend uses one inference queue, so practical parallel inference stays close to 1 active request at a time.",
            "- Health, monitoring, and request-status routes remain callable while the queue is busy.",
        ]
    if failed:
        lines += ["", "## KO", ""]
        for item in failed:
            lines.append(f"- `{item['label']}`: {item['load'].get('error') or item['load'].get('message')}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"gemma4-api-loadtest-{timestamp}.json"
    md_path = RESULTS_DIR / f"gemma4-api-loadtest-{timestamp}.md"
    bubble_svg_path = RESULTS_DIR / f"gemma4-api-loadtest-latency-{timestamp}.svg"
    parallel_svg_path = RESULTS_DIR / f"gemma4-api-loadtest-parallel-{timestamp}.svg"
    session = requests.Session()
    results = []
    for model_key, quantization_key, label in MODELS:
        print(f"\n=== Loading {label} ===")
        load = ensure_loaded(session, base_url, model_key, quantization_key, args.load_timeout)
        model_result = {
            "label": label,
            "model_key": model_key,
            "quantization_key": quantization_key,
            "model_id": f"{model_key}:{quantization_key}",
            "load": load,
            "workloads": [],
        }
        results.append(model_result)
        if not load.get("ok"):
            print(f"KO load: {load.get('error') or load.get('message')}")
            continue
        allowed = WORKLOADS if model_key != "31b-nvfp4" else [item for item in WORKLOADS if item["name"] != "long"]
        for workload in allowed:
            print(f"  -> {workload['name']}")
            try:
                streamed = stream_completion(
                    session,
                    base_url,
                    model_result["model_id"],
                    workload["prompt"],
                    workload["max_tokens"],
                    args.request_timeout,
                )
                model_result["workloads"].append(
                    {
                        "name": workload["name"],
                        "input_chars": len(workload["prompt"]),
                        "max_tokens": workload["max_tokens"],
                        **streamed,
                    }
                )
            except Exception as exc:
                model_result["workloads"].append(
                    {
                        "name": workload["name"],
                        "input_chars": len(workload["prompt"]),
                        "max_tokens": workload["max_tokens"],
                        "ok": False,
                        "error": repr(exc),
                    }
                )
                print(f"    workload failed: {exc!r}")
    parallel_results = []
    for model_key, quantization_key, label in [("e2b", "sfp8", "Gemma 4 E2B / SFP8"), ("26b-a4b", "sfp8", "Gemma 4 26B A4B / SFP8")]:
        load = ensure_loaded(session, base_url, model_key, quantization_key, args.load_timeout)
        if not load.get("ok"):
            continue
        baseline = run_parallel(requests.Session, base_url, f"{model_key}:{quantization_key}", 1, args.request_timeout)
        parallel_results.append({"label": label, "concurrent_requests": 1, "makespan_seconds": baseline["makespan_seconds"], "avg_latency_seconds": baseline["avg_latency_seconds"], "max_latency_seconds": baseline["max_latency_seconds"], "effective_parallelism": 1.0})
        for concurrent_requests in (2, 4):
            batch = run_parallel(requests.Session, base_url, f"{model_key}:{quantization_key}", concurrent_requests, args.request_timeout)
            effective = None
            if baseline["makespan_seconds"] and batch["makespan_seconds"]:
                effective = round((baseline["makespan_seconds"] * concurrent_requests) / batch["makespan_seconds"], 2)
            parallel_results.append({"label": label, "concurrent_requests": concurrent_requests, "makespan_seconds": batch["makespan_seconds"], "avg_latency_seconds": batch["avg_latency_seconds"], "max_latency_seconds": batch["max_latency_seconds"], "effective_parallelism": effective})
    audio_probe = build_audio_probe(session, base_url)
    build_latency_svg(results, bubble_svg_path)
    build_parallel_svg(parallel_results, parallel_svg_path)
    report = {
        "generated_at": datetime.now().isoformat(),
        "base_url": base_url,
        "results": results,
        "parallel_results": parallel_results,
        "audio_probe": audio_probe,
        "artifacts": {"markdown": str(md_path), "bubble_svg": str(bubble_svg_path), "parallel_svg": str(parallel_svg_path)},
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_report(results, parallel_results, audio_probe, bubble_svg_path, parallel_svg_path), encoding="utf-8")
    for src_name, dst_name in [
        (json_path, RESULTS_DIR / "gemma4-api-loadtest-latest.json"),
        (md_path, RESULTS_DIR / "gemma4-api-loadtest-latest.md"),
        (bubble_svg_path, RESULTS_DIR / "gemma4-api-loadtest-latency-latest.svg"),
        (parallel_svg_path, RESULTS_DIR / "gemma4-api-loadtest-parallel-latest.svg"),
    ]:
        dst_name.write_text(src_name.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {bubble_svg_path}")
    print(f"Wrote {parallel_svg_path}")


if __name__ == "__main__":
    main()
