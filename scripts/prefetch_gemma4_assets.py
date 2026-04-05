from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from piper.download_voices import download_voice


BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / ".hf-cache"
VOICE_DIR = BASE_DIR / "tts" / "voices"
VOICE_NAME = "fr_FR-siwis-medium"

BF16_REPOS = [
    ("Gemma 4 E2B BF16", "google/gemma-4-E2B-it"),
    ("Gemma 4 E4B BF16", "google/gemma-4-E4B-it"),
    ("Gemma 4 26B A4B BF16", "google/gemma-4-26B-A4B-it"),
    ("Gemma 4 31B BF16", "google/gemma-4-31B-it"),
]

GGUF_TARGETS = [
    ("Gemma 4 E2B Q4_0", "bartowski/google_gemma-4-E2B-it-GGUF", "Q4_0"),
    ("Gemma 4 E2B Q8_0", "ggml-org/gemma-4-E2B-it-GGUF", "Q8_0"),
    ("Gemma 4 E4B Q4_0", "bartowski/google_gemma-4-E4B-it-GGUF", "Q4_0"),
    ("Gemma 4 E4B Q8_0", "ggml-org/gemma-4-E4B-it-GGUF", "Q8_0"),
    ("Gemma 4 26B A4B Q4_0", "bartowski/google_gemma-4-26B-A4B-it-GGUF", "Q4_0"),
    ("Gemma 4 26B A4B Q8_0", "ggml-org/gemma-4-26B-A4B-it-GGUF", "Q8_0"),
    ("Gemma 4 31B Q4_0", "bartowski/google_gemma-4-31B-it-GGUF", "Q4_0"),
    ("Gemma 4 31B Q8_0", "ggml-org/gemma-4-31B-it-GGUF", "Q8_0"),
]

NVFP4_REPOS = [
    ("Gemma 4 31B IT NVFP4", "nvidia/Gemma-4-31B-IT-NVFP4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch Gemma 4 checkpoints, GGUF files, and local TTS assets."
    )
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-nvfp4", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[prefetch] {message}", flush=True)


def select_gguf_filename(api: HfApi, repo_id: str, quantization: str) -> str:
    quant_lower = quantization.lower()
    candidates = [
        path
        for path in api.list_repo_files(repo_id=repo_id, repo_type="model")
        if path.lower().endswith(".gguf") and quant_lower in Path(path).name.lower()
    ]
    if not candidates:
        raise RuntimeError(f"No GGUF file matching {quantization} was found in {repo_id}.")

    def rank(path: str) -> tuple[int, int, str]:
        filename = Path(path).name.lower()
        penalty = 0
        if "imat" in filename:
            penalty += 10
        return (penalty, len(filename), filename)

    return sorted(candidates, key=rank)[0]


def prefetch_snapshot(label: str, repo_id: str) -> None:
    print_step(f"Downloading {label} from {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=CACHE_DIR,
    )


def prefetch_gguf(label: str, api: HfApi, repo_id: str, quantization: str) -> None:
    filename = select_gguf_filename(api, repo_id, quantization)
    print_step(f"Downloading {label} -> {repo_id}/{filename}")
    hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=filename,
        cache_dir=CACHE_DIR,
    )


def prefetch_tts() -> None:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    print_step(f"Downloading Piper voice {VOICE_NAME}")
    download_voice(VOICE_NAME, VOICE_DIR)


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    if not args.skip_bf16:
        for label, repo_id in BF16_REPOS:
            prefetch_snapshot(label, repo_id)

    if not args.skip_gguf:
        for label, repo_id, quantization in GGUF_TARGETS:
            prefetch_gguf(label, api, repo_id, quantization)

    if not args.skip_nvfp4:
        for label, repo_id in NVFP4_REPOS:
            prefetch_snapshot(label, repo_id)

    if not args.skip_tts:
        prefetch_tts()

    print_step(f"Done. Shared cache is ready in {CACHE_DIR}")


if __name__ == "__main__":
    main()
