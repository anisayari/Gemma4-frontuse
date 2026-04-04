from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app import CACHE_DIR, MODEL_SPECS, get_gpu_total_memory_gib, get_windows_commit_snapshot


RESULTS_DIR = BASE_DIR / "benchmark" / "results"
DEFAULT_PROMPT = (
    "Explain in a compact technical paragraph why local multimodal inference "
    "throughput matters for a desktop copilot."
)
DEFAULT_SYSTEM_PROMPT = "You are a concise technical assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Gemma 4 text-generation throughput on the local machine."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[spec["key"] for spec in MODEL_SPECS],
        help="Model keys to benchmark. Defaults to all Gemma 4 variants known by the backend.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Measured decode length in generated tokens.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=32,
        help="Warmup decode length before timing.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of measured runs after warmup.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        default=True,
        help="Use only files already present in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Permit missing files to download from Hugging Face.",
    )
    parser.add_argument(
        "--output-prefix",
        default="gemma4-5090-benchmark",
        help="Prefix for JSON and Markdown result files.",
    )
    return parser.parse_args()


def resolve_model_specs(model_keys: list[str]) -> list[dict]:
    specs_by_key = {spec["key"]: spec for spec in MODEL_SPECS}
    resolved = []
    for key in model_keys:
        normalized = key.strip().lower()
        spec = specs_by_key.get(normalized)
        if spec is None:
            raise SystemExit(f"Unknown model key: {key}")
        resolved.append(spec)
    return resolved


def pick_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for device in model.hf_device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
    return model.device


def make_prompt_text(processor, prompt: str, system_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def maybe_sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def unload_model(processor, model) -> None:
    del processor
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_error(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def run_generate(model, processor, prompt_text: str, max_new_tokens: int) -> dict:
    device = pick_input_device(model)
    inputs = processor(text=prompt_text, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    maybe_sync_cuda()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    maybe_sync_cuda()
    elapsed = time.perf_counter() - started
    output_tokens = int(outputs.shape[-1])
    generated_tokens = output_tokens - input_tokens
    tokens_per_second = generated_tokens / elapsed if elapsed > 0 else 0.0

    result = {
        "elapsed_seconds": round(elapsed, 4),
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
    }
    if torch.cuda.is_available():
        result["peak_vram_allocated_gib"] = round(
            torch.cuda.max_memory_allocated() / (1024**3), 2
        )
        result["peak_vram_reserved_gib"] = round(
            torch.cuda.max_memory_reserved() / (1024**3), 2
        )
    return result


def benchmark_one_model(spec: dict, args: argparse.Namespace) -> dict:
    record: dict = {
        "model_key": spec["key"],
        "label": spec["label"],
        "hf_model_id": spec["hf_model_id"],
        "architecture": spec["architecture"],
        "tier": spec["tier"],
        "quantization": "bf16",
        "official_bf16_memory_gib": spec["memory_requirements_gib"]["bf16"],
        "gpu_total_memory_gib": get_gpu_total_memory_gib(),
        "windows_commit_before": get_windows_commit_snapshot(),
    }

    if not torch.cuda.is_available():
        record["status"] = "error"
        record["error"] = "CUDA is not available on this machine."
        return record

    processor = None
    model = None
    try:
        load_started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            spec["hf_model_id"],
            cache_dir=str(CACHE_DIR),
            local_files_only=not args.allow_downloads,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            spec["hf_model_id"],
            cache_dir=str(CACHE_DIR),
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=not args.allow_downloads,
        )
        maybe_sync_cuda()
        record["load_seconds"] = round(time.perf_counter() - load_started, 4)
        record["status"] = "loaded"
        record["device_map"] = getattr(model, "hf_device_map", None)
        record["vram_allocated_after_load_gib"] = round(
            torch.cuda.memory_allocated() / (1024**3), 2
        )
        record["vram_reserved_after_load_gib"] = round(
            torch.cuda.memory_reserved() / (1024**3), 2
        )

        prompt_text = make_prompt_text(processor, DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT)
        record["warmup"] = run_generate(model, processor, prompt_text, args.warmup_tokens)
        measured_runs = [
            run_generate(model, processor, prompt_text, args.max_new_tokens)
            for _ in range(args.runs)
        ]
        record["runs"] = measured_runs
        speeds = [run["tokens_per_second"] for run in measured_runs]
        elapsed = [run["elapsed_seconds"] for run in measured_runs]
        record["summary"] = {
            "median_tokens_per_second": round(sorted(speeds)[len(speeds) // 2], 2),
            "mean_tokens_per_second": round(sum(speeds) / len(speeds), 2),
            "best_tokens_per_second": round(max(speeds), 2),
            "worst_tokens_per_second": round(min(speeds), 2),
            "mean_elapsed_seconds": round(sum(elapsed) / len(elapsed), 4),
            "generated_tokens_per_run": args.max_new_tokens,
        }
        record["status"] = "ok"
        record["windows_commit_after"] = get_windows_commit_snapshot()
        return record
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = format_error(exc)
        record["windows_commit_after"] = get_windows_commit_snapshot()
        return record
    finally:
        if processor is not None and model is not None:
            unload_model(processor, model)
        elif model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def read_nvidia_smi() -> dict | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    line = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return {"raw": line}
    return {
        "name": parts[0],
        "driver_version": parts[1],
        "memory_total_mib": int(parts[2]),
    }


def build_markdown_report(payload: dict) -> str:
    lines = [
        "# Gemma 4 Benchmark",
        "",
        f"- Timestamp UTC: `{payload['timestamp_utc']}`",
        f"- GPU: `{payload['hardware'].get('name', 'unknown')}`",
        f"- Driver: `{payload['hardware'].get('driver_version', 'unknown')}`",
        f"- GPU total memory: `{payload['hardware'].get('memory_total_mib', 'unknown')} MiB`",
        f"- Max new tokens per measured run: `{payload['config']['max_new_tokens']}`",
        f"- Measured runs per model: `{payload['config']['runs']}`",
        "",
        "| Model | Status | BF16 official memory | Mean tok/s | Best tok/s | Notes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]

    for result in payload["results"]:
        if result["status"] == "ok":
            summary = result["summary"]
            notes = (
                f"load {result['load_seconds']} s, load VRAM {result['vram_reserved_after_load_gib']} GiB"
            )
            lines.append(
                f"| {result['label']} | ok | {result['official_bf16_memory_gib']:.1f} GiB | "
                f"{summary['mean_tokens_per_second']:.2f} | {summary['best_tokens_per_second']:.2f} | {notes} |"
            )
        else:
            error = result.get("error", "unknown failure").replace("|", "/")
            lines.append(
                f"| {result['label']} | {result['status']} | {result['official_bf16_memory_gib']:.1f} GiB | "
                f"- | - | {error} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    os_local_only = not args.allow_downloads
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": read_nvidia_smi() or {},
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "runs": args.runs,
            "local_only": os_local_only,
            "cache_dir": str(CACHE_DIR),
            "prompt": DEFAULT_PROMPT,
        },
        "results": [],
    }

    for spec in resolve_model_specs(args.models):
        print(f"[benchmark] {spec['label']} ({spec['hf_model_id']})")
        record = benchmark_one_model(spec, args)
        payload["results"].append(record)
        print(json.dumps(record, indent=2))

    timestamp_slug = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"{args.output_prefix}-{timestamp_slug}.json"
    md_path = RESULTS_DIR / f"{args.output_prefix}-{timestamp_slug}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_report(payload), encoding="utf-8")

    latest_json = RESULTS_DIR / f"{args.output_prefix}-latest.json"
    latest_md = RESULTS_DIR / f"{args.output_prefix}-latest.md"
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_md.write_text(build_markdown_report(payload), encoding="utf-8")

    print(f"[benchmark] wrote {json_path}")
    print(f"[benchmark] wrote {md_path}")


if __name__ == "__main__":
    main()
