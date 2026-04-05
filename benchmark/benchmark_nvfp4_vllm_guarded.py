from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "benchmark" / "results"
LOG_DIR = BASE_DIR / "logs"
HF_CACHE_DIR = BASE_DIR / ".hf-cache"
RUN_LOG_PATH = LOG_DIR / "nvfp4-benchmark-run.log"
WATCHDOG_LOG_PATH = LOG_DIR / "nvfp4-benchmark-watchdog.log"
LATEST_RESULT_PATH = RESULTS_DIR / "gemma4-nvfp4-vllm-benchmark-latest.json"
LATEST_SUMMARY_PATH = RESULTS_DIR / "gemma4-nvfp4-vllm-summary-latest.md"
BENCHMARK_SCRIPT = BASE_DIR / "benchmark" / "benchmark_nvfp4_vllm.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NVFP4 vLLM benchmark with Windows-side timeout and process watchdogs."
    )
    parser.add_argument("--wsl-distro", default=os.getenv("GEMMA4_VLLM_WSL_DISTRO", "Ubuntu"))
    parser.add_argument("--wsl-activate", default=os.getenv("GEMMA4_VLLM_WSL_ACTIVATE", "~/vllm-gemma4/bin/activate"))
    parser.add_argument("--total-timeout-seconds", type=int, default=1200)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--no-enforce-eager", action="store_true")
    return parser.parse_args()


def to_wsl_path(path: Path) -> str:
    normalized = str(path.resolve()).replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        return f"/mnt/{normalized[0].lower()}{normalized[2:]}"
    return normalized


def expand_wsl_home(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("~/"):
        return "${HOME}/" + normalized[2:]
    return normalized


def bash_double_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def log_watchdog(event: str, **payload: object) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with WATCHDOG_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_wsl_command(distro: str, script: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["wsl", "-d", distro, "bash", "-lc", script],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def cleanup_wsl_processes(distro: str) -> None:
    cleanup_script = (
        "pkill -f 'benchmark_nvfp4_vllm.py' >/dev/null 2>&1 || true; "
        "pkill -f 'vllm serve nvidia/Gemma-4-31B-IT-NVFP4' >/dev/null 2>&1 || true; "
        "pkill -f 'VLLM::EngineCore' >/dev/null 2>&1 || true"
    )
    try:
        completed = run_wsl_command(distro, cleanup_script, timeout=45)
        log_watchdog(
            "cleanup",
            distro=distro,
            returncode=completed.returncode,
            stdout_tail=completed.stdout[-400:],
            stderr_tail=completed.stderr[-400:],
        )
    except Exception as exc:
        log_watchdog("cleanup_error", distro=distro, error=f"{exc.__class__.__name__}: {exc}")


def query_gpu_snapshot() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        return {"status": "error", "detail": completed.stderr.strip() or completed.stdout.strip()}

    line = (completed.stdout.strip().splitlines() or [""])[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 4:
        return {"status": "error", "detail": line}
    memory_used_mib = int(parts[1])
    memory_total_mib = int(parts[2])
    gpu_utilization_percent = int(parts[3])
    if (
        memory_used_mib < 0
        or memory_total_mib <= 0
        or memory_used_mib > memory_total_mib * 2
        or gpu_utilization_percent < 0
        or gpu_utilization_percent > 100
    ):
        return {"status": "error", "detail": line}
    return {
        "status": "ok",
        "name": parts[0],
        "memory_used_mib": memory_used_mib,
        "memory_total_mib": memory_total_mib,
        "gpu_utilization_percent": gpu_utilization_percent,
    }


def build_summary(result: dict) -> str:
    config = result.get("config") or {}
    if result.get("status") == "ok":
        return "\n".join(
            [
                "# Gemma 4 31B IT NVFP4 Benchmark",
                "",
                f"- Model: `{result['model_id']}`",
                "- Runtime: `vLLM` on `WSL Ubuntu`",
                "- Quantization: `NVFP4 / ModelOpt FP4`",
                f"- Load time: `{result.get('load_seconds', 'n/a')} s`",
                f"- Generation throughput: `{result.get('tokens_per_second', 'n/a')} tok/s`",
                f"- Generated tokens: `{result.get('generated_tokens', 'n/a')}`",
                "",
                "## Config",
                "",
                f"- `max_model_len={config.get('max_model_len')}`",
                f"- `gpu_memory_utilization={config.get('gpu_memory_utilization')}`",
                f"- `max_tokens={config.get('max_tokens')}`",
                f"- `enforce_eager={config.get('enforce_eager')}`",
                f"- `cpu_offload_gb={config.get('cpu_offload_gb')}`",
                "",
            ]
        )

    return "\n".join(
        [
            "# Gemma 4 31B IT NVFP4 Benchmark",
            "",
            "- Status: `failed`",
            f"- Error: `{result.get('error', 'unknown error')}`",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_mtime_before = (
        LATEST_RESULT_PATH.stat().st_mtime if LATEST_RESULT_PATH.exists() else 0.0
    )
    RUN_LOG_PATH.write_text("", encoding="utf-8")

    cleanup_wsl_processes(args.wsl_distro)

    benchmark_script_wsl = shlex.quote(to_wsl_path(BENCHMARK_SCRIPT))
    bash_command = " ".join(
        [
            "set -euo pipefail;",
            f"source {bash_double_quote(expand_wsl_home(args.wsl_activate))};",
            f"export HF_HOME={bash_double_quote(to_wsl_path(HF_CACHE_DIR))};",
            f"export HUGGINGFACE_HUB_CACHE={bash_double_quote(to_wsl_path(HF_CACHE_DIR))};",
            "export VLLM_NVFP4_GEMM_BACKEND=cutlass;",
            f"export NVFP4_MAX_MODEL_LEN={args.max_model_len};",
            f"export NVFP4_GPU_MEMORY_UTILIZATION={args.gpu_memory_utilization};",
            f"export NVFP4_MAX_TOKENS={args.max_tokens};",
            f"export NVFP4_ENFORCE_EAGER={0 if args.no_enforce_eager else 1};",
            f"export NVFP4_CPU_OFFLOAD_GB={args.cpu_offload_gb};",
            f"timeout {max(60, args.total_timeout_seconds - 15)}s python {benchmark_script_wsl}",
        ]
    )

    log_watchdog(
        "start",
        distro=args.wsl_distro,
        total_timeout_seconds=args.total_timeout_seconds,
        poll_seconds=args.poll_seconds,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=args.max_tokens,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=not args.no_enforce_eager,
        gpu_snapshot=query_gpu_snapshot(),
    )

    with RUN_LOG_PATH.open("a", encoding="utf-8") as run_log:
        process = subprocess.Popen(
            ["wsl", "-d", args.wsl_distro, "bash", "-lc", bash_command],
            cwd=str(BASE_DIR),
            stdout=run_log,
            stderr=run_log,
            text=True,
        )

    started = time.time()
    timed_out = False

    try:
        while process.poll() is None:
            elapsed = time.time() - started
            log_watchdog(
                "poll",
                elapsed_seconds=round(elapsed, 1),
                gpu_snapshot=query_gpu_snapshot(),
            )
            if elapsed > args.total_timeout_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                break
            time.sleep(max(1, args.poll_seconds))
    finally:
        cleanup_wsl_processes(args.wsl_distro)

    if timed_out:
        tail = RUN_LOG_PATH.read_text(encoding="utf-8", errors="replace")[-3000:]
        log_watchdog("timeout", run_log_tail=tail)
        raise SystemExit(
            "NVFP4 benchmark timed out and the WSL processes were cleaned up. "
            f"See {RUN_LOG_PATH} and {WATCHDOG_LOG_PATH}."
        )

    if process.returncode != 0:
        tail = RUN_LOG_PATH.read_text(encoding="utf-8", errors="replace")[-3000:]
        log_watchdog("failed", returncode=process.returncode, run_log_tail=tail)
        raise SystemExit(
            f"NVFP4 benchmark failed with exit code {process.returncode}. "
            f"See {RUN_LOG_PATH} and {WATCHDOG_LOG_PATH}."
        )

    if not LATEST_RESULT_PATH.exists() or LATEST_RESULT_PATH.stat().st_mtime <= result_mtime_before:
        tail = RUN_LOG_PATH.read_text(encoding="utf-8", errors="replace")[-3000:]
        log_watchdog("missing_result", run_log_tail=tail)
        raise SystemExit(
            "The guarded NVFP4 benchmark finished without refreshing the latest JSON result."
        )

    result = json.loads(LATEST_RESULT_PATH.read_text(encoding="utf-8"))
    LATEST_SUMMARY_PATH.write_text(build_summary(result), encoding="utf-8")
    log_watchdog("done", result_status=result.get("status"), gpu_snapshot=query_gpu_snapshot())

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[nvfp4-guarded] wrote {LATEST_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
