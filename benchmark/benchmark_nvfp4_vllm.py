from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


RESULTS_DIR = Path("/mnt/c/Users/Anis AYARI/Desktop/projects/gemma4-test/benchmark/results")
MODEL_ID = "nvidia/Gemma-4-31B-IT-NVFP4"
PROMPT = (
    "Explain in one compact technical paragraph why local multimodal inference "
    "throughput matters for a workstation copilot."
)
SYSTEM_PROMPT = "You are a concise technical assistant."
MAX_MODEL_LEN = int(os.getenv("NVFP4_MAX_MODEL_LEN", "1024"))
GPU_MEMORY_UTILIZATION = float(os.getenv("NVFP4_GPU_MEMORY_UTILIZATION", "0.99"))
MAX_TOKENS = int(os.getenv("NVFP4_MAX_TOKENS", "128"))
ENFORCE_EAGER = os.getenv("NVFP4_ENFORCE_EAGER", "1") == "1"
CPU_OFFLOAD_GB = float(os.getenv("NVFP4_CPU_OFFLOAD_GB", "0"))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "runtime": "vllm",
        "quantization": "modelopt_fp4",
        "status": "starting",
        "platform": "WSL Ubuntu",
        "config": {
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "enforce_eager": ENFORCE_EAGER,
            "cpu_offload_gb": CPU_OFFLOAD_GB,
        },
    }

    timestamp_slug = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_path = RESULTS_DIR / f"gemma4-nvfp4-vllm-benchmark-{timestamp_slug}.json"
    latest_path = RESULTS_DIR / "gemma4-nvfp4-vllm-benchmark-latest.json"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        chat_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        load_started = time.perf_counter()
        llm = LLM(
            model=MODEL_ID,
            quantization="modelopt_fp4",
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=ENFORCE_EAGER,
            cpu_offload_gb=CPU_OFFLOAD_GB,
        )
        payload["load_seconds"] = round(time.perf_counter() - load_started, 4)

        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        started = time.perf_counter()
        outputs = llm.generate([chat_prompt], sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - started

        output = outputs[0].outputs[0]
        generated_tokens = len(output.token_ids)
        payload["status"] = "ok"
        payload["elapsed_seconds"] = round(elapsed, 4)
        payload["generated_tokens"] = generated_tokens
        payload["tokens_per_second"] = round(
            generated_tokens / elapsed if elapsed else 0.0, 2
        )
        payload["text_preview"] = output.text[:400]
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{exc.__class__.__name__}: {exc}"

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    result_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
