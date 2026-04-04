from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "benchmark" / "results"
LLAMA_BENCH = BASE_DIR / "tools" / "llama.cpp" / "bin" / "llama-bench.exe"
HF_HOME = BASE_DIR / ".hf-cache"


@dataclass(frozen=True)
class BenchTarget:
    model_key: str
    label: str
    quantization: str
    repo_id: str
    official: bool


TARGETS = [
    BenchTarget("e2b", "Gemma 4 E2B", "Q8_0", "ggml-org/gemma-4-E2B-it-GGUF", True),
    BenchTarget("e2b", "Gemma 4 E2B", "Q4_0", "bartowski/google_gemma-4-E2B-it-GGUF", False),
    BenchTarget("e4b", "Gemma 4 E4B", "Q8_0", "ggml-org/gemma-4-E4B-it-GGUF", True),
    BenchTarget("e4b", "Gemma 4 E4B", "Q4_0", "bartowski/google_gemma-4-E4B-it-GGUF", False),
    BenchTarget("26b-a4b", "Gemma 4 26B A4B", "Q8_0", "ggml-org/gemma-4-26B-A4B-it-GGUF", True),
    BenchTarget("26b-a4b", "Gemma 4 26B A4B", "Q4_0", "bartowski/google_gemma-4-26B-A4B-it-GGUF", False),
    BenchTarget("31b", "Gemma 4 31B", "Q8_0", "ggml-org/gemma-4-31B-it-GGUF", True),
    BenchTarget("31b", "Gemma 4 31B", "Q4_0", "bartowski/google_gemma-4-31B-it-GGUF", False),
]


def parse_bench_json(stdout: str) -> list[dict]:
    start = stdout.find("[")
    if start < 0:
        raise ValueError("llama-bench did not emit JSON output.")
    payload, _ = json.JSONDecoder().raw_decode(stdout[start:])
    return payload


def pick_row(rows: list[dict], *, n_prompt: int, n_gen: int) -> dict | None:
    for row in rows:
        if row.get("n_prompt") == n_prompt and row.get("n_gen") == n_gen:
            return row
    return None


def build_command(target: BenchTarget) -> list[str]:
    return [
        str(LLAMA_BENCH),
        "-o",
        "json",
        "-r",
        "2",
        "-p",
        "256",
        "-n",
        "128",
        "-fa",
        "1",
        "-ngl",
        "999",
        "-hf",
        f"{target.repo_id}:{target.quantization}",
    ]


def run_target(target: BenchTarget) -> dict:
    env = dict(os.environ)
    env["HF_HOME"] = str(HF_HOME)
    started = datetime.now(timezone.utc).isoformat()
    command = build_command(target)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(BASE_DIR),
    )
    record = {
        "started_utc": started,
        "model_key": target.model_key,
        "label": target.label,
        "quantization": target.quantization,
        "repo_id": target.repo_id,
        "repo_source": "official" if target.official else "community",
        "command": command,
        "returncode": completed.returncode,
    }

    if completed.returncode != 0:
        record["status"] = "failed"
        record["stderr"] = completed.stderr[-4000:]
        record["stdout_tail"] = completed.stdout[-4000:]
        return record

    try:
        rows = parse_bench_json(completed.stdout)
    except Exception as exc:
        record["status"] = "failed"
        record["parse_error"] = f"{exc.__class__.__name__}: {exc}"
        record["stdout_tail"] = completed.stdout[-4000:]
        return record

    pp_row = pick_row(rows, n_prompt=256, n_gen=0)
    tg_row = pick_row(rows, n_prompt=0, n_gen=128)
    record["status"] = "ok"
    record["model_filename"] = tg_row.get("model_filename") if tg_row else None
    record["model_type"] = tg_row.get("model_type") if tg_row else None
    record["model_size_bytes"] = tg_row.get("model_size") if tg_row else None
    record["gpu_info"] = tg_row.get("gpu_info") if tg_row else None
    record["prompt_tokens_per_second"] = pp_row.get("avg_ts") if pp_row else None
    record["prompt_stddev_tokens_per_second"] = pp_row.get("stddev_ts") if pp_row else None
    record["gen_tokens_per_second"] = tg_row.get("avg_ts") if tg_row else None
    record["gen_stddev_tokens_per_second"] = tg_row.get("stddev_ts") if tg_row else None
    record["samples_gen_tokens_per_second"] = tg_row.get("samples_ts") if tg_row else None
    record["raw_rows"] = rows
    return record


def build_markdown(results: list[dict]) -> str:
    lines = [
        "# Gemma 4 GGUF Benchmark",
        "",
        "- Runtime: `llama.cpp` CUDA",
        "- Prompt benchmark: `256` prompt tokens",
        "- Generation benchmark: `128` generated tokens",
        "- Repetitions: `2`",
        "",
        "| Model | Quant | Source | Status | Prompt tok/s | Gen tok/s | Stddev |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]

    for result in results:
        if result["status"] == "ok":
            lines.append(
                f"| {result['label']} | {result['quantization']} | {result['repo_source']} | ok | "
                f"{result['prompt_tokens_per_second']:.2f} | {result['gen_tokens_per_second']:.2f} | "
                f"{result['gen_stddev_tokens_per_second']:.2f} |"
            )
        else:
            note = result.get("parse_error") or result.get("stderr", "").splitlines()[-1]
            note = (note or "failed").replace("|", "/")
            lines.append(
                f"| {result['label']} | {result['quantization']} | {result['repo_source']} | failed | - | - | - |"
            )
            lines.append("")
            lines.append(f"- `{result['label']} {result['quantization']}`: {note}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not LLAMA_BENCH.exists():
        raise SystemExit(f"Missing llama-bench executable: {LLAMA_BENCH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for target in TARGETS:
        print(f"[gguf-bench] {target.label} {target.quantization} ({target.repo_id})", flush=True)
        results.append(run_target(target))

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "llama_bench": str(LLAMA_BENCH),
        "results": results,
    }
    timestamp_slug = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"gemma4-gguf-benchmark-{timestamp_slug}.json"
    md_path = RESULTS_DIR / f"gemma4-gguf-benchmark-{timestamp_slug}.md"
    latest_json = RESULTS_DIR / "gemma4-gguf-benchmark-latest.json"
    latest_md = RESULTS_DIR / "gemma4-gguf-benchmark-latest.md"

    json_text = json.dumps(payload, indent=2)
    md_text = build_markdown(results)
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    print(f"[gguf-bench] wrote {json_path}")
    print(f"[gguf-bench] wrote {md_path}")


if __name__ == "__main__":
    main()
