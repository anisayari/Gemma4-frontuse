from __future__ import annotations

import base64
import ctypes
import gc
import io
import json
import logging
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
import urllib.error
import urllib.request
import wave
import socket
from queue import Queue
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig
from piper.download_voices import download_voice
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForMultimodalLM, AutoProcessor, TextIteratorStreamer


BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "web" / "dist"
CACHE_DIR = Path(os.getenv("HF_HOME", BASE_DIR / ".hf-cache"))
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "gemma4-lab.log"
LLAMA_SERVER_LOG_PATH = LOG_DIR / "llama-server.log"
WSL_VLLM_LOG_PATH = LOG_DIR / "vllm-wsl.log"
TTS_DIR = BASE_DIR / "tts"
TTS_VOICE_DIR = TTS_DIR / "voices"
TTS_GENERATED_DIR = TTS_DIR / "generated"
LLAMA_SERVER_BIN = BASE_DIR / "tools" / "llama.cpp" / "bin" / "llama-server.exe"
LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8011
LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
WSL_VLLM_DISTRO = os.getenv("GEMMA4_VLLM_WSL_DISTRO", "Ubuntu")
WSL_VLLM_HOST = "127.0.0.1"
WSL_VLLM_PORT = int(os.getenv("GEMMA4_VLLM_PORT", "8012"))
WSL_VLLM_URL = f"http://{WSL_VLLM_HOST}:{WSL_VLLM_PORT}"
WSL_VLLM_ACTIVATE = os.getenv(
    "GEMMA4_VLLM_WSL_ACTIVATE",
    "~/vllm-gemma4/bin/activate",
)
WSL_VLLM_STARTUP_TIMEOUT_SECONDS = int(
    os.getenv("GEMMA4_VLLM_STARTUP_TIMEOUT_SECONDS", "480")
)
WSL_VLLM_MAX_MODEL_LEN = int(os.getenv("GEMMA4_VLLM_MAX_MODEL_LEN", "256"))
WSL_VLLM_GPU_MEMORY_UTILIZATION = float(
    os.getenv("GEMMA4_VLLM_GPU_MEMORY_UTILIZATION", "0.94")
)
WSL_VLLM_MAX_NUM_SEQS = int(os.getenv("GEMMA4_VLLM_MAX_NUM_SEQS", "1"))
WSL_VLLM_MAX_NUM_BATCHED_TOKENS = int(
    os.getenv("GEMMA4_VLLM_MAX_NUM_BATCHED_TOKENS", "128")
)
WSL_VLLM_CPU_OFFLOAD_GB = float(os.getenv("GEMMA4_VLLM_CPU_OFFLOAD_GB", "0"))
WSL_VLLM_MAX_COMPLETION_TOKENS = int(
    os.getenv("GEMMA4_VLLM_MAX_COMPLETION_TOKENS", "64")
)
WSL_VLLM_APPROX_CHARS_PER_TOKEN = int(
    os.getenv("GEMMA4_VLLM_APPROX_CHARS_PER_TOKEN", "4")
)
WSL_VLLM_IMAGE_TOKEN_BUDGET = int(
    os.getenv("GEMMA4_VLLM_IMAGE_TOKEN_BUDGET", "64")
)
WSL_VLLM_MESSAGE_OVERHEAD_CHARS = int(
    os.getenv("GEMMA4_VLLM_MESSAGE_OVERHEAD_CHARS", "24")
)
WSL_VLLM_SYSTEM_PROMPT_MAX_CHARS = int(
    os.getenv("GEMMA4_VLLM_SYSTEM_PROMPT_MAX_CHARS", "220")
)
WSL_VLLM_REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("GEMMA4_VLLM_REQUEST_TIMEOUT_SECONDS", "360")
)
WSL_VLLM_LOG_STALL_TIMEOUT_SECONDS = int(
    os.getenv("GEMMA4_VLLM_LOG_STALL_TIMEOUT_SECONDS", "180")
)
LOCAL_FILES_ONLY = os.getenv("GEMMA4_LOCAL_ONLY", "0") == "1"
DEFAULT_MODEL_KEY = os.getenv("GEMMA4_MODEL_KEY", "e4b")
DEFAULT_QUANTIZATION_KEY = os.getenv("GEMMA4_QUANTIZATION_KEY", "bf16")
DEFAULT_TTS_VOICE = os.getenv("GEMMA4_TTS_VOICE", "fr_FR-siwis-medium")
DEFAULT_SYSTEM_PROMPT = (
    "You are Gemma 4 running locally on a workstation. Be concise, technical, and "
    "explicit about what can be inferred from the provided media."
)
DOCS_NOTE = (
    "Model modalities here follow the official Gemma 4 Supported Modalities tables. "
    "For safety, this UI only exposes text, image, and audio inputs; audio is limited "
    "to E2B and E4B."
)
QUANTIZATION_NOTE = (
    "Quantization memory estimates follow the official Google Gemma 4 overview page "
    "(updated 2026-04-02). This local backend runs the official Hugging Face "
    "checkpoints in BF16, routes Q4_0 through a local llama.cpp runtime, maps the "
    "Google SFP8 planning slot to a practical local Q8_0 GGUF path, and can route "
    "NVIDIA's NVFP4 checkpoint through WSL vLLM."
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
TTS_VOICE_DIR.mkdir(parents=True, exist_ok=True)
TTS_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

CLIP_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
WHITESPACE_PATTERN = re.compile(r"\s+")

logger = logging.getLogger("gemma4_lab")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

MODEL_SPECS = [
    {
        "key": "e2b",
        "label": "Gemma 4 E2B",
        "hf_model_id": "google/gemma-4-E2B-it",
        "architecture": "Dense",
        "tier": "Edge",
        "context_length": "128K",
        "parameter_summary": "2.3B effective / 5.1B with embeddings",
        "active_parameter_summary": "2.3B effective",
        "supported_modalities": ["text", "image", "audio"],
        "supports_audio": True,
        "supports_image": True,
        "supports_text": True,
        "memory_requirements_gib": {"bf16": 9.6, "sfp8": 4.6, "q4_0": 3.2},
        "llama_cpp_hf_repo_ids": {
            "q4_0": "bartowski/google_gemma-4-E2B-it-GGUF",
            "sfp8": "ggml-org/gemma-4-E2B-it-GGUF",
        },
        "doc_summary": "Small edge model with native audio support.",
    },
    {
        "key": "e4b",
        "label": "Gemma 4 E4B",
        "hf_model_id": "google/gemma-4-E4B-it",
        "architecture": "Dense",
        "tier": "Edge",
        "context_length": "128K",
        "parameter_summary": "4.5B effective / 8B with embeddings",
        "active_parameter_summary": "4.5B effective",
        "supported_modalities": ["text", "image", "audio"],
        "supports_audio": True,
        "supports_image": True,
        "supports_text": True,
        "memory_requirements_gib": {"bf16": 15.0, "sfp8": 7.5, "q4_0": 5.0},
        "llama_cpp_hf_repo_ids": {
            "q4_0": "bartowski/google_gemma-4-E4B-it-GGUF",
            "sfp8": "ggml-org/gemma-4-E4B-it-GGUF",
        },
        "doc_summary": "Best balanced small model with native audio support.",
    },
    {
        "key": "26b-a4b",
        "label": "Gemma 4 26B A4B",
        "hf_model_id": "google/gemma-4-26B-A4B-it",
        "architecture": "MoE",
        "tier": "Workstation",
        "context_length": "256K",
        "parameter_summary": "25.2B total / 3.8B active",
        "active_parameter_summary": "3.8B active",
        "supported_modalities": ["text", "image"],
        "supports_audio": False,
        "supports_image": True,
        "supports_text": True,
        "memory_requirements_gib": {"bf16": 48.0, "sfp8": 25.0, "q4_0": 15.6},
        "llama_cpp_hf_repo_ids": {
            "q4_0": "bartowski/google_gemma-4-26B-A4B-it-GGUF",
            "sfp8": "ggml-org/gemma-4-26B-A4B-it-GGUF",
        },
        "doc_summary": "Mixture-of-Experts variant tuned for faster workstation inference.",
    },
    {
        "key": "31b",
        "label": "Gemma 4 31B",
        "hf_model_id": "google/gemma-4-31B-it",
        "architecture": "Dense",
        "tier": "Workstation",
        "context_length": "256K",
        "parameter_summary": "30.7B dense",
        "active_parameter_summary": "30.7B active",
        "supported_modalities": ["text", "image"],
        "supports_audio": False,
        "supports_image": True,
        "supports_text": True,
        "memory_requirements_gib": {"bf16": 58.3, "sfp8": 30.4, "q4_0": 17.4},
        "min_windows_commit_available_gib": 64.0,
        "llama_cpp_hf_repo_ids": {
            "q4_0": "bartowski/google_gemma-4-31B-it-GGUF",
            "sfp8": "ggml-org/gemma-4-31B-it-GGUF",
        },
        "doc_summary": "Largest dense Gemma 4 variant for local workstation use.",
    },
    {
        "key": "31b-nvfp4",
        "label": "Gemma 4 31B IT NVFP4",
        "hf_model_id": "nvidia/Gemma-4-31B-IT-NVFP4",
        "architecture": "Dense",
        "tier": "Blackwell",
        "context_length": "256K",
        "parameter_summary": "30.7B dense / NVIDIA ModelOpt NVFP4",
        "active_parameter_summary": "30.7B active",
        "supported_modalities": ["text", "image", "video"],
        "supports_audio": False,
        "supports_image": True,
        "supports_text": True,
        "memory_requirements_gib": {"nvfp4": None},
        "llama_cpp_hf_repo_ids": {},
        "doc_summary": (
            "NVIDIA-optimized NVFP4 checkpoint. Inputs are text, image, and video; output "
            "is text. This local lab can route it through an experimental WSL vLLM path on "
            "NVIDIA Blackwell."
        ),
    },
]
MODEL_SPECS_BY_KEY = {spec["key"]: spec for spec in MODEL_SPECS}

QUANTIZATION_SPECS = [
    {
        "key": "bf16",
        "label": "BF16",
        "precision_bits": 16,
        "runtime_supported": True,
        "status": "available",
        "runtime_family": "transformers",
        "doc_summary": "Default 16-bit Hugging Face path used by this local backend.",
    },
    {
        "key": "sfp8",
        "label": "SFP8",
        "precision_bits": 8,
        "runtime_supported": True,
        "status": "llama.cpp",
        "runtime_family": "llama.cpp",
        "hf_file_label": "Q8_0",
        "doc_summary": (
            "The Google SFP8 slot is mapped here to a practical local Q8_0 GGUF "
            "runtime through llama.cpp."
        ),
    },
    {
        "key": "q4_0",
        "label": "Q4_0",
        "precision_bits": 4,
        "runtime_supported": True,
        "status": "llama.cpp",
        "runtime_family": "llama.cpp",
        "doc_summary": "Quantized local runtime served through llama.cpp. In this app build, Q4_0 is enabled for fast text chat.",
    },
    {
        "key": "nvfp4",
        "label": "NVFP4",
        "precision_bits": 4,
        "runtime_supported": True,
        "status": "wsl-vllm",
        "runtime_family": "vllm-wsl",
        "doc_summary": (
            "NVIDIA ModelOpt NVFP4 checkpoint format. The official model card targets "
            "vLLM on NVIDIA Blackwell under Linux. This lab bridges it through a local "
            "WSL vLLM runtime with tuned memory settings."
        ),
    },
]
QUANTIZATION_SPECS_BY_KEY = {spec["key"]: spec for spec in QUANTIZATION_SPECS}

if DEFAULT_MODEL_KEY not in MODEL_SPECS_BY_KEY:
    DEFAULT_MODEL_KEY = "e4b"

if DEFAULT_QUANTIZATION_KEY not in QUANTIZATION_SPECS_BY_KEY:
    DEFAULT_QUANTIZATION_KEY = "bf16"


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ModelLoadRequest(BaseModel):
    model_key: str
    quantization_key: str = DEFAULT_QUANTIZATION_KEY


def get_model_spec(model_key: str | None) -> dict:
    key = (model_key or DEFAULT_MODEL_KEY).strip().lower()
    spec = MODEL_SPECS_BY_KEY.get(key)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown model key: {model_key}")
    return spec


def get_quantization_spec(quantization_key: str | None) -> dict:
    key = (quantization_key or DEFAULT_QUANTIZATION_KEY).strip().lower()
    spec = QUANTIZATION_SPECS_BY_KEY.get(key)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown quantization key: {quantization_key}")
    return spec


def serialize_model_spec(spec: dict) -> dict:
    return {
        "key": spec["key"],
        "label": spec["label"],
        "hf_model_id": spec["hf_model_id"],
        "architecture": spec["architecture"],
        "tier": spec["tier"],
        "context_length": spec["context_length"],
        "parameter_summary": spec["parameter_summary"],
        "active_parameter_summary": spec["active_parameter_summary"],
        "supported_modalities": spec["supported_modalities"],
        "supports_audio": spec["supports_audio"],
        "supports_image": spec["supports_image"],
        "supports_text": spec["supports_text"],
        "memory_requirements_gib": spec["memory_requirements_gib"],
        "llama_cpp_hf_repo_ids": spec.get("llama_cpp_hf_repo_ids", {}),
        "doc_summary": spec["doc_summary"],
    }


def serialize_quantization_spec(spec: dict) -> dict:
    return {
        "key": spec["key"],
        "label": spec["label"],
        "precision_bits": spec["precision_bits"],
        "runtime_supported": spec["runtime_supported"],
        "status": spec["status"],
        "runtime_family": spec.get("runtime_family", "planning"),
        "doc_summary": spec["doc_summary"],
    }


def get_gpu_total_memory_gib() -> float | None:
    if not torch.cuda.is_available():
        return None

    return round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def bytes_to_gib(value: int) -> float:
    return round(value / (1024**3), 2)


def get_windows_commit_snapshot() -> dict | None:
    if platform.system() != "Windows":
        return None

    snapshot = MEMORYSTATUSEX()
    snapshot.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(snapshot)):
        return None

    return {
        "total_physical_gib": bytes_to_gib(snapshot.ullTotalPhys),
        "available_physical_gib": bytes_to_gib(snapshot.ullAvailPhys),
        "commit_limit_gib": bytes_to_gib(snapshot.ullTotalPageFile),
        "available_commit_gib": bytes_to_gib(snapshot.ullAvailPageFile),
    }


def preflight_model_load(spec: dict) -> str | None:
    required_commit_gib = spec.get("min_windows_commit_available_gib")
    if required_commit_gib is None:
        return None

    snapshot = get_windows_commit_snapshot()
    if snapshot is None:
        return None

    if snapshot["available_commit_gib"] < required_commit_gib:
        return (
            f"{spec['label']} was blocked before loading because Windows only has "
            f"{snapshot['available_commit_gib']:.2f} GiB of commit memory available, and "
            f"this model needs about {required_commit_gib:.0f} GiB free to map its bf16 "
            "weights reliably. Increase the Windows page file and retry, or switch to "
            "Gemma 4 E4B / 26B A4B."
        )

    return None


def preflight_quantization_support(model_spec: dict, quantization_spec: dict) -> str | None:
    if quantization_spec["runtime_supported"]:
        return None

    if quantization_spec["key"] not in model_spec["memory_requirements_gib"]:
        return f"{quantization_spec['label']} is not available for {model_spec['label']}."

    if quantization_spec["key"] == "nvfp4":
        return (
            f"{model_spec['label']} uses NVIDIA's NVFP4 checkpoint format, but this local "
            "backend cannot load it here. The official model card targets vLLM with "
            "ModelOpt on NVIDIA Blackwell under Linux."
        )

    memory_estimate = model_spec["memory_requirements_gib"].get(quantization_spec["key"])
    return (
        f"{model_spec['label']} with {quantization_spec['label']} is exposed in the UI using "
        f"Google's memory estimate ({memory_estimate:.1f} GiB), but this local backend cannot "
        "load it yet. The current app supports BF16 through Transformers and Q4_0 through "
        "llama.cpp. This quantization still needs a different runtime or checkpoint format."
    )


def render_model_load_error(spec: dict, exc: Exception) -> str:
    raw_message = str(exc).strip() or exc.__class__.__name__
    lowered = raw_message.lower()

    if "os error 1455" in lowered or "fichier de pagination" in lowered:
        return (
            f"{spec['label']} could not be loaded on this Windows setup because the paging "
            "file is too small for the model weights (os error 1455). Increase the Windows "
            "page file or switch to Gemma 4 E4B / 26B A4B."
        )

    if "out of memory" in lowered or "cuda" in lowered and "memory" in lowered:
        return (
            f"{spec['label']} could not be loaded because the machine ran out of memory. "
            "Free RAM or VRAM, or switch to a smaller Gemma 4 variant."
        )

    if "no available memory for the cache blocks" in lowered:
        return (
            f"{spec['label']} loaded its weights, but there was not enough remaining VRAM "
            "for the KV cache. Reduce the configured context length, free more GPU memory, "
            "or switch to a lighter runtime path."
        )

    if "less than desired gpu memory utilization" in lowered:
        return (
            f"{spec['label']} could not start because the GPU already had too little free "
            "memory for the configured vLLM reservation. Close other GPU workloads and retry."
        )

    if "cannot re-initialize the input batch when cpu weight offloading is enabled" in lowered:
        return (
            f"{spec['label']} hit a current vLLM limitation while CPU weight offloading was "
            "enabled. Retry with CPU offload disabled for this model."
        )

    if "engine core initialization failed" in lowered:
        return (
            f"{spec['label']} failed after the weights were loaded but before the vLLM engine "
            "finished initializing. Check the WSL vLLM log for the root cause."
        )

    return f"{spec['label']} failed to load: {raw_message}"


def parse_openai_error_detail(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()

    return text


def render_vllm_request_error(spec: dict, detail: str) -> str:
    lowered = detail.lower()
    if "maximum context length is" in lowered:
        return (
            f"{spec['label']} is running in a compact {WSL_VLLM_MAX_MODEL_LEN}-token "
            "WSL vLLM profile on this machine. The lab trims history and caps the "
            f"reply to about {WSL_VLLM_MAX_COMPLETION_TOKENS} new tokens, but this turn "
            "still overflowed the context window. Shorten the prompt or clear older turns."
        )
    return detail


def to_wsl_path(path: Path) -> str:
    normalized = str(path.resolve()).replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        return f"/mnt/{normalized[0].lower()}{normalized[2:]}"
    return normalized


def read_text_tail(path: Path, *, max_chars: int = 2400) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:].strip()


def expand_wsl_home(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("~/"):
        return "${HOME}/" + normalized[2:]
    return normalized


def bash_double_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def hf_cache_repo_dir(repo_id: str, *, use_hub: bool = False) -> Path:
    cache_root = CACHE_DIR / "hub" if use_hub else CACHE_DIR
    return cache_root / f"models--{repo_id.replace('/', '--')}"


def get_listening_pids_for_port(port: int) -> set[int]:
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        logger.exception("Failed to inspect TCP listeners for port=%s", port)
        return set()

    stdout_text = result.stdout or ""
    pids: set[int] = set()
    needle = f":{port}"
    for line in stdout_text.splitlines():
        if needle not in line or "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local_address = parts[1]
        state = parts[3]
        pid_text = parts[4]
        if not local_address.endswith(needle) or state != "LISTENING":
            continue
        try:
            pids.add(int(pid_text))
        except ValueError:
            continue
    return pids


def kill_process_ids(pids: set[int], *, exclude_pid: int | None = None) -> None:
    for pid in sorted(pids):
        if exclude_pid is not None and pid == exclude_pid:
            continue
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
        except Exception:
            logger.exception("Failed to terminate stale process pid=%s", pid)


def ensure_llama_cpp_repo_cache(repo_id: str) -> None:
    direct_repo_dir = hf_cache_repo_dir(repo_id)
    hub_repo_dir = hf_cache_repo_dir(repo_id, use_hub=True)
    if not hub_repo_dir.exists():
        return

    if direct_repo_dir.exists():
        if (direct_repo_dir / "snapshots").exists() and (direct_repo_dir / "refs").exists():
            return
        shutil.rmtree(direct_repo_dir, ignore_errors=True)

    direct_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    junction = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(direct_repo_dir), str(hub_repo_dir)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if junction.returncode == 0 and direct_repo_dir.exists():
        return

    shutil.copytree(hub_repo_dir, direct_repo_dir, symlinks=True)


def pil_image_to_data_url(image: Image.Image, *, quality: int = 92) -> str:
    buffer = io.BytesIO()
    normalized = image.convert("RGB")
    normalized.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def normalize_tts_text(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return ""

    normalized = CODE_BLOCK_PATTERN.sub(" Code block omitted from voice output. ", normalized)
    normalized = MARKDOWN_LINK_PATTERN.sub(r"\1", normalized)
    normalized = INLINE_CODE_PATTERN.sub(r"\1", normalized)
    normalized = normalized.replace("|", ", ")
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def sanitize_llama_reply(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"<\|?channel\|?>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(thought|analysis)\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


class LocalTTSService:
    def __init__(self) -> None:
        self._voice = None
        self._load_lock = threading.Lock()
        self._synthesis_lock = threading.Lock()
        self._voice_name = DEFAULT_TTS_VOICE
        self._sample_rate = None

    @property
    def model_path(self) -> Path:
        return TTS_VOICE_DIR / f"{self._voice_name}.onnx"

    @property
    def config_path(self) -> Path:
        return TTS_VOICE_DIR / f"{self._voice_name}.onnx.json"

    def _ensure_assets(self) -> None:
        if self.model_path.exists() and self.config_path.exists():
            return

        logger.info("Downloading TTS voice=%s into %s", self._voice_name, TTS_VOICE_DIR)
        download_voice(self._voice_name, TTS_VOICE_DIR)

    def ensure_loaded(self) -> PiperVoice:
        if self._voice is not None:
            return self._voice

        with self._load_lock:
            if self._voice is None:
                self._ensure_assets()
                logger.info("Loading TTS voice=%s", self._voice_name)
                self._voice = PiperVoice.load(
                    self.model_path,
                    config_path=self.config_path,
                    use_cuda=False,
                    download_dir=TTS_VOICE_DIR,
                )
                self._sample_rate = getattr(self._voice.config, "sample_rate", None)

        return self._voice

    def cleanup_generated_clips(self, *, max_files: int = 48, max_age_seconds: int = 6 * 60 * 60) -> None:
        now = time.time()
        clips = sorted(
            TTS_GENERATED_DIR.glob("*.wav"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for index, clip_path in enumerate(clips):
            try:
                should_delete = index >= max_files or (now - clip_path.stat().st_mtime) > max_age_seconds
            except FileNotFoundError:
                continue

            if should_delete:
                clip_path.unlink(missing_ok=True)

    def synthesize(self, text: str) -> dict | None:
        normalized_text = normalize_tts_text(text)
        if not normalized_text:
            return None

        voice = self.ensure_loaded()
        clip_id = uuid.uuid4().hex
        clip_path = TTS_GENERATED_DIR / f"{clip_id}.wav"
        synthesis_config = SynthesisConfig(
            length_scale=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            normalize_audio=True,
            volume=1.0,
        )

        with self._synthesis_lock:
            start_time = time.perf_counter()
            with clip_path.open("wb") as raw_file:
                with wave.open(raw_file, "wb") as wav_file:
                    voice.synthesize_wav(normalized_text, wav_file, synthesis_config)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        clip_size = clip_path.stat().st_size
        self.cleanup_generated_clips()
        logger.info(
            "Generated TTS clip voice=%s clip_id=%s bytes=%s elapsed_ms=%s",
            self._voice_name,
            clip_id,
            clip_size,
            elapsed_ms,
        )
        return {
            "clip_id": clip_id,
            "url": f"/api/tts/clips/{clip_id}.wav",
            "mime_type": "audio/wav",
            "voice": self._voice_name,
            "sample_rate": self._sample_rate,
            "elapsed_ms": elapsed_ms,
            "size_bytes": clip_size,
        }

    def health(self) -> dict:
        return {
            "enabled": True,
            "voice": self._voice_name,
            "loaded": self._voice is not None,
            "sample_rate": self._sample_rate,
            "voice_model_path": str(self.model_path),
            "clips_dir": str(TTS_GENERATED_DIR),
        }


class LlamaCppServerRuntime:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._current_model_key: str | None = None
        self._current_quantization_key: str | None = None

    def _cleanup_stale_processes(self, *, exclude_pid: int | None = None) -> None:
        kill_process_ids(get_listening_pids_for_port(LLAMA_SERVER_PORT), exclude_pid=exclude_pid)

    @property
    def is_loaded(self) -> bool:
        return (
            self._process is not None
            and self._process.poll() is None
            and self._current_model_key is not None
            and self._current_quantization_key is not None
        )

    def matches(self, model_key: str, quantization_key: str) -> bool:
        return (
            self.is_loaded
            and self._current_model_key == model_key
            and self._current_quantization_key == quantization_key
        )

    def unload(self) -> None:
        with self._process_lock:
            process = self._process
            self._process = None
            self._current_model_key = None
            self._current_quantization_key = None

        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

        self._cleanup_stale_processes()

    def _runtime_hf_label(self, quantization_spec: dict) -> str:
        return str(quantization_spec.get("hf_file_label", quantization_spec["label"]))

    def _gpu_layers_for(self, spec: dict, quantization_spec: dict) -> str:
        if spec["key"] == "31b" and quantization_spec["key"] == "sfp8":
            return "56"
        return "999"

    def _build_command(self, spec: dict, quantization_spec: dict) -> list[str]:
        repo_id = spec.get("llama_cpp_hf_repo_ids", {}).get(quantization_spec["key"])
        if not repo_id:
            raise HTTPException(
                status_code=501,
                detail=(
                    f"{spec['label']} in {quantization_spec['label']} does not have a "
                    "configured llama.cpp checkpoint in this app."
                ),
            )

        ensure_llama_cpp_repo_cache(repo_id)

        return [
            str(LLAMA_SERVER_BIN),
            "--hf-repo",
            f"{repo_id}:{self._runtime_hf_label(quantization_spec)}",
            "-ngl",
            self._gpu_layers_for(spec, quantization_spec),
            "--host",
            LLAMA_SERVER_HOST,
            "--port",
            str(LLAMA_SERVER_PORT),
            "--parallel",
            "1",
            "--slots",
            "--reasoning",
            "off",
            "--reasoning-format",
            "none",
            "-c",
            "4096",
        ]

    def _wait_until_ready(self, *, timeout_seconds: int = 300) -> None:
        deadline = time.time() + timeout_seconds
        last_error = "llama.cpp server did not become ready."
        while time.time() < deadline:
            with self._process_lock:
                process = self._process

            if process is None:
                raise RuntimeError("llama.cpp server process was not started.")

            if process.poll() is not None:
                raise RuntimeError("llama.cpp server exited before becoming ready.")

            try:
                request = urllib.request.Request(f"{LLAMA_SERVER_URL}/health", method="GET")
                with urllib.request.urlopen(request, timeout=5) as response:
                    payload = json.load(response)
                if response.status == 200 and payload.get("status") == "ok":
                    return
            except Exception as exc:  # pragma: no cover - transient readiness loop
                last_error = str(exc)

            time.sleep(1)

        raise RuntimeError(last_error)

    def load(
        self,
        spec: dict,
        quantization_spec: dict,
        *,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> None:
        def emit(progress: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(progress, message)

        if self.matches(spec["key"], quantization_spec["key"]):
            emit(100, f"{spec['label']} is already loaded in {quantization_spec['label']}.")
            return

        emit(18, "Preparing llama.cpp quantized runtime...")
        if not LLAMA_SERVER_BIN.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Missing llama.cpp server binary: {LLAMA_SERVER_BIN}",
            )

        emit(32, "Releasing the previous quantized runtime...")
        self.unload()
        self._cleanup_stale_processes()
        command = self._build_command(spec, quantization_spec)
        emit(48, "Launching llama.cpp server...")
        env = dict(os.environ)
        env["HF_HOME"] = str(CACHE_DIR)
        env["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR)

        with LLAMA_SERVER_LOG_PATH.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(BASE_DIR),
                stdout=log_file,
                stderr=log_file,
                text=True,
                env=env,
            )

        with self._process_lock:
            self._process = process
            self._current_model_key = spec["key"]
            self._current_quantization_key = quantization_spec["key"]

        emit(76, "Waiting for llama.cpp to finish loading the quantized model...")
        self._wait_until_ready()
        emit(100, f"{spec['label']} is ready in {quantization_spec['label']}.")

    def _request_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{LLAMA_SERVER_URL}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore").strip() or str(exc)
            raise HTTPException(status_code=exc.code, detail=detail) from exc
        except urllib.error.URLError as exc:
            raise HTTPException(
                status_code=503,
                detail="The llama.cpp runtime is unavailable on localhost.",
            ) from exc

    def _build_messages(
        self,
        *,
        prompt: str,
        system_prompt: str,
        history: list[HistoryTurn],
    ) -> list[dict]:
        messages: list[dict] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        *,
        spec: dict,
        quantization_spec: dict,
        prompt: str,
        system_prompt: str,
        history: list[HistoryTurn],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        started = time.perf_counter()
        payload = {
            "messages": self._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
            ),
            "stream": False,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        response_payload = self._request_json("/v1/chat/completions", payload)
        choice = (response_payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = response_payload.get("usage") or {}
        timings = response_payload.get("timings") or {}
        reply = sanitize_llama_reply(message.get("content", ""))
        return {
            "reply": reply,
            "thought": message.get("reasoning_content"),
            "raw_response": message.get("content", ""),
            "parsed": {"role": "assistant", "content": reply},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": usage.get("completion_tokens"),
            "active_model_key": spec["key"],
            "active_model": serialize_model_spec(spec),
            "active_quantization_key": quantization_spec["key"],
            "active_quantization": serialize_quantization_spec(quantization_spec),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_allocated_gib": 0.0,
            "vram_reserved_gib": 0.0,
            "timings": timings,
        }

    def stream_generate(
        self,
        *,
        spec: dict,
        quantization_spec: dict,
        prompt: str,
        system_prompt: str,
        history: list[HistoryTurn],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        tts_enabled: bool,
    ):
        payload = {
            "messages": self._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
            ),
            "stream": True,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        request = urllib.request.Request(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        reply_chunks: list[str] = []
        thought_chunks: list[str] = []
        started = time.perf_counter()
        emitted_reply = ""

        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                yield {"event": "start"}
                for raw_line in response:
                    line = raw_line.decode("utf-8", "ignore").strip()
                    if not line or not line.startswith("data: "):
                        continue

                    payload_line = line[6:].strip()
                    if payload_line == "[DONE]":
                        break

                    event_payload = json.loads(payload_line)
                    choice = (event_payload.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text = delta["content"]
                        reply_chunks.append(text)
                        cleaned_reply = sanitize_llama_reply("".join(reply_chunks))
                        if len(cleaned_reply) > len(emitted_reply):
                            delta_text = cleaned_reply[len(emitted_reply) :]
                            emitted_reply = cleaned_reply
                            if delta_text:
                                yield {"event": "token", "text": delta_text}
                    if delta.get("reasoning_content"):
                        thought_chunks.append(delta["reasoning_content"])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore").strip() or str(exc)
            raise HTTPException(status_code=exc.code, detail=detail) from exc
        except urllib.error.URLError as exc:
            raise HTTPException(
                status_code=503,
                detail="The llama.cpp runtime is unavailable on localhost.",
            ) from exc

        reply = sanitize_llama_reply("".join(reply_chunks))
        thought = "".join(thought_chunks).strip() or None
        response_payload = {
            "reply": reply,
            "thought": thought,
            "raw_response": reply,
            "parsed": {"role": "assistant", "content": reply},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "prompt_tokens": None,
            "generated_tokens": None,
            "active_model_key": spec["key"],
            "active_model": serialize_model_spec(spec),
            "active_quantization_key": quantization_spec["key"],
            "active_quantization": serialize_quantization_spec(quantization_spec),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_allocated_gib": 0.0,
            "vram_reserved_gib": 0.0,
        }
        if tts_enabled and reply:
            try:
                response_payload["tts_audio"] = tts_service.synthesize(reply)
            except Exception:
                logger.exception(
                    "TTS synthesis failed after llama.cpp streaming key=%s quantization=%s",
                    spec["key"],
                    quantization_spec["key"],
                )
                response_payload["tts_audio"] = None
                response_payload["tts_audio_error"] = "Local TTS synthesis failed."
        yield {"event": "done", "payload": response_payload}


class WslVllmServerRuntime:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._current_model_key: str | None = None
        self._current_quantization_key: str | None = None

    @property
    def is_loaded(self) -> bool:
        return (
            self._process is not None
            and self._process.poll() is None
            and self._current_model_key is not None
            and self._current_quantization_key is not None
        )

    def matches(self, model_key: str, quantization_key: str) -> bool:
        return (
            self.is_loaded
            and self._current_model_key == model_key
            and self._current_quantization_key == quantization_key
        )

    def _cleanup_wsl_processes(self) -> None:
        cleanup_script = (
            "pkill -f 'benchmark_nvfp4_vllm.py' >/dev/null 2>&1 || true; "
            "pkill -f 'vllm serve nvidia/Gemma-4-31B-IT-NVFP4' >/dev/null 2>&1 || true; "
            f"pkill -f '--port {WSL_VLLM_PORT}' >/dev/null 2>&1 || true; "
            "pkill -f 'VLLM::EngineCore' >/dev/null 2>&1 || true"
        )
        try:
            subprocess.run(
                ["wsl", "-d", WSL_VLLM_DISTRO, "bash", "-lc", cleanup_script],
                cwd=str(BASE_DIR),
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            logger.exception("WSL vLLM cleanup command failed")

    def unload(self) -> None:
        with self._process_lock:
            process = self._process
            self._process = None
            self._current_model_key = None
            self._current_quantization_key = None

        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

        self._cleanup_wsl_processes()

    def _build_command(self, spec: dict) -> list[str]:
        if spec["key"] != "31b-nvfp4":
            raise HTTPException(
                status_code=501,
                detail=f"{spec['label']} does not have a configured WSL vLLM runtime.",
            )

        log_path_wsl = shlex.quote(to_wsl_path(WSL_VLLM_LOG_PATH))
        model_id = shlex.quote(spec["hf_model_id"])
        serve_command = " ".join(
            [
                "vllm",
                "serve",
                model_id,
                "--host",
                WSL_VLLM_HOST,
                "--port",
                str(WSL_VLLM_PORT),
                "--trust-remote-code",
                "--quantization",
                "modelopt",
                "--max-model-len",
                str(WSL_VLLM_MAX_MODEL_LEN),
                "--gpu-memory-utilization",
                str(WSL_VLLM_GPU_MEMORY_UTILIZATION),
                "--max-num-seqs",
                str(WSL_VLLM_MAX_NUM_SEQS),
                "--max-num-batched-tokens",
                str(WSL_VLLM_MAX_NUM_BATCHED_TOKENS),
                "--cpu-offload-gb",
                str(WSL_VLLM_CPU_OFFLOAD_GB),
                "--uvicorn-log-level",
                "warning",
                "--enforce-eager",
            ]
        )
        bash_command = (
            "set -euo pipefail; "
            f"source {bash_double_quote(expand_wsl_home(WSL_VLLM_ACTIVATE))}; "
            "export VLLM_NVFP4_GEMM_BACKEND=cutlass; "
            "export VLLM_WORKER_MULTIPROC_METHOD=spawn; "
            f"{serve_command} >> {log_path_wsl} 2>&1"
        )
        return ["wsl", "-d", WSL_VLLM_DISTRO, "bash", "-lc", bash_command]

    def _wait_until_ready(
        self,
        *,
        timeout_seconds: int = WSL_VLLM_STARTUP_TIMEOUT_SECONDS,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> None:
        deadline = time.time() + timeout_seconds
        started = time.time()
        last_error = "WSL vLLM server did not become ready."
        last_log_mtime = (
            WSL_VLLM_LOG_PATH.stat().st_mtime if WSL_VLLM_LOG_PATH.exists() else None
        )
        last_log_progress_at = time.time()

        while time.time() < deadline:
            with self._process_lock:
                process = self._process

            if process is None:
                raise RuntimeError("WSL vLLM process was not started.")

            if process.poll() is not None:
                log_tail = read_text_tail(WSL_VLLM_LOG_PATH)
                raise RuntimeError(log_tail or "WSL vLLM exited before becoming ready.")

            if WSL_VLLM_LOG_PATH.exists():
                try:
                    current_log_mtime = WSL_VLLM_LOG_PATH.stat().st_mtime
                except OSError:
                    current_log_mtime = last_log_mtime
                if current_log_mtime != last_log_mtime:
                    last_log_mtime = current_log_mtime
                    last_log_progress_at = time.time()

            elapsed = time.time() - started
            if progress_callback is not None:
                if elapsed < 45:
                    progress_callback(58, "Booting the WSL vLLM runtime...")
                elif elapsed < 150:
                    progress_callback(72, "Loading NVFP4 weights and warming kernels...")
                else:
                    progress_callback(88, "Finalizing KV cache and chat server startup...")

            if (time.time() - last_log_progress_at) > WSL_VLLM_LOG_STALL_TIMEOUT_SECONDS:
                log_tail = read_text_tail(WSL_VLLM_LOG_PATH)
                raise RuntimeError(
                    log_tail
                    or (
                        "The WSL vLLM startup log stopped changing for too long. "
                        "The runtime was treated as stalled and will be reset."
                    )
                )

            try:
                request = urllib.request.Request(f"{WSL_VLLM_URL}/health", method="GET")
                with urllib.request.urlopen(request, timeout=5) as response:
                    response.read()
                if response.status == 200:
                    return
            except Exception as exc:  # pragma: no cover - transient readiness loop
                last_error = str(exc)

            time.sleep(2)

        log_tail = read_text_tail(WSL_VLLM_LOG_PATH)
        raise RuntimeError(log_tail or last_error)

    def load(
        self,
        spec: dict,
        quantization_spec: dict,
        *,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> None:
        def emit(progress: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(progress, message)

        if self.matches(spec["key"], quantization_spec["key"]):
            emit(100, f"{spec['label']} is already loaded in {quantization_spec['label']}.")
            return

        emit(12, "Preparing the WSL vLLM runtime...")
        self.unload()
        emit(24, "Cleaning stale WSL vLLM processes...")
        self._cleanup_wsl_processes()
        command = self._build_command(spec)
        emit(36, "Launching the NVIDIA NVFP4 server inside WSL...")

        with WSL_VLLM_LOG_PATH.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(BASE_DIR),
                stdout=log_file,
                stderr=log_file,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

        with self._process_lock:
            self._process = process
            self._current_model_key = spec["key"]
            self._current_quantization_key = quantization_spec["key"]

        self._wait_until_ready(progress_callback=emit)
        emit(100, f"{spec['label']} is ready in {quantization_spec['label']}.")

    def _request_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{WSL_VLLM_URL}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=WSL_VLLM_REQUEST_TIMEOUT_SECONDS
            ) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raw_detail = exc.read().decode("utf-8", "ignore").strip() or str(exc)
            detail = render_vllm_request_error(
                get_model_spec(self._current_model_key or "31b-nvfp4"),
                parse_openai_error_detail(raw_detail),
            )
            raise HTTPException(status_code=exc.code, detail=detail) from exc
        except TimeoutError as exc:
            self.unload()
            raise HTTPException(
                status_code=504,
                detail=(
                    "The WSL vLLM runtime timed out while generating a reply and was "
                    "reset to avoid leaving a stuck GPU process behind."
                ),
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout) or "timed out" in str(
                exc.reason
            ).lower():
                self.unload()
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "The WSL vLLM runtime timed out while generating a reply and was "
                        "reset to avoid leaving a stuck GPU process behind."
                    ),
                ) from exc
            raise HTTPException(
                status_code=503,
                detail="The WSL vLLM runtime is unavailable on localhost.",
            ) from exc

    def _build_messages(
        self,
        *,
        prompt: str,
        system_prompt: str,
        history: list[HistoryTurn],
        image: Image.Image | None,
    ) -> list[dict]:
        messages: list[dict] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})

        if image is None:
            messages.append({"role": "user", "content": prompt})
            return messages

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_image_to_data_url(image)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return messages

    def _trim_text(self, text: str, max_chars: int, *, preserve_tail: bool = False) -> str:
        normalized = text.strip()
        if max_chars <= 0:
            return ""
        if len(normalized) <= max_chars:
            return normalized
        if max_chars <= 1:
            return normalized[:max_chars]
        if preserve_tail and max_chars > 3:
            return "..." + normalized[-(max_chars - 3) :]
        return normalized[: max_chars - 1].rstrip() + "..."

    def _shape_request_payload(
        self,
        *,
        prompt: str,
        system_prompt: str,
        history: list[HistoryTurn],
        image: Image.Image | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        effective_max_tokens = min(
            max(16, int(max_new_tokens)),
            WSL_VLLM_MAX_COMPLETION_TOKENS,
            max(16, WSL_VLLM_MAX_MODEL_LEN - 32),
        )
        input_budget_tokens = max(
            48,
            WSL_VLLM_MAX_MODEL_LEN
            - effective_max_tokens
            - (WSL_VLLM_IMAGE_TOKEN_BUDGET if image is not None else 0),
        )
        input_budget_chars = input_budget_tokens * WSL_VLLM_APPROX_CHARS_PER_TOKEN

        trimmed_system_prompt = self._trim_text(
            system_prompt,
            min(WSL_VLLM_SYSTEM_PROMPT_MAX_CHARS, max(80, input_budget_chars // 4)),
        )
        remaining_for_prompt = max(
            160,
            input_budget_chars
            - len(trimmed_system_prompt)
            - (WSL_VLLM_MESSAGE_OVERHEAD_CHARS * 2),
        )
        trimmed_prompt = self._trim_text(prompt, remaining_for_prompt)

        remaining_history_chars = max(
            0,
            input_budget_chars
            - len(trimmed_system_prompt)
            - len(trimmed_prompt)
            - (WSL_VLLM_MESSAGE_OVERHEAD_CHARS * 2),
        )
        shaped_history: list[HistoryTurn] = []
        for turn in reversed(history):
            content = turn.content.strip()
            if not content:
                continue
            turn_budget = len(content) + WSL_VLLM_MESSAGE_OVERHEAD_CHARS
            if turn_budget <= remaining_history_chars:
                shaped_history.append(HistoryTurn(role=turn.role, content=content))
                remaining_history_chars -= turn_budget
                continue
            if not shaped_history and remaining_history_chars > (
                WSL_VLLM_MESSAGE_OVERHEAD_CHARS + 64
            ):
                truncated_content = self._trim_text(
                    content,
                    remaining_history_chars - WSL_VLLM_MESSAGE_OVERHEAD_CHARS,
                    preserve_tail=True,
                )
                if truncated_content:
                    shaped_history.append(
                        HistoryTurn(role=turn.role, content=truncated_content)
                    )
            break

        shaped_history.reverse()
        return {
            "messages": self._build_messages(
                prompt=trimmed_prompt,
                system_prompt=trimmed_system_prompt,
                history=shaped_history,
                image=image,
            ),
            "stream": False,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

    def generate(
        self,
        *,
        spec: dict,
        quantization_spec: dict,
        prompt: str,
        system_prompt: str,
        history: list[HistoryTurn],
        image: Image.Image | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        started = time.perf_counter()
        payload = self._shape_request_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            image=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        response_payload = self._request_json("/v1/chat/completions", payload)
        choice = (response_payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = response_payload.get("usage") or {}
        return {
            "reply": message.get("content", ""),
            "thought": message.get("reasoning_content"),
            "raw_response": message.get("content", ""),
            "parsed": {"role": "assistant", "content": message.get("content", "")},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": usage.get("completion_tokens"),
            "active_model_key": spec["key"],
            "active_model": serialize_model_spec(spec),
            "active_quantization_key": quantization_spec["key"],
            "active_quantization": serialize_quantization_spec(quantization_spec),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_allocated_gib": 0.0,
            "vram_reserved_gib": 0.0,
        }

    def stream_generate(
        self,
        *,
        spec: dict,
        quantization_spec: dict,
        prompt: str,
        system_prompt: str,
        history: list[HistoryTurn],
        image: Image.Image | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        tts_enabled: bool,
    ):
        payload = self._shape_request_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            image=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        payload["stream"] = True
        request = urllib.request.Request(
            f"{WSL_VLLM_URL}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        reply_chunks: list[str] = []
        thought_chunks: list[str] = []
        started = time.perf_counter()

        try:
            with urllib.request.urlopen(
                request, timeout=WSL_VLLM_REQUEST_TIMEOUT_SECONDS
            ) as response:
                yield {"event": "start"}
                for raw_line in response:
                    line = raw_line.decode("utf-8", "ignore").strip()
                    if not line or not line.startswith("data: "):
                        continue

                    payload_line = line[6:].strip()
                    if payload_line == "[DONE]":
                        break

                    event_payload = json.loads(payload_line)
                    choice = (event_payload.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text = delta["content"]
                        reply_chunks.append(text)
                        yield {"event": "token", "text": text}
                    if delta.get("reasoning_content"):
                        thought_chunks.append(delta["reasoning_content"])
        except urllib.error.HTTPError as exc:
            raw_detail = exc.read().decode("utf-8", "ignore").strip() or str(exc)
            detail = render_vllm_request_error(spec, parse_openai_error_detail(raw_detail))
            raise HTTPException(status_code=exc.code, detail=detail) from exc
        except TimeoutError as exc:
            self.unload()
            raise HTTPException(
                status_code=504,
                detail=(
                    "The WSL vLLM runtime timed out while streaming a reply and was "
                    "reset to avoid leaving a stuck GPU process behind."
                ),
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout) or "timed out" in str(
                exc.reason
            ).lower():
                self.unload()
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "The WSL vLLM runtime timed out while streaming a reply and was "
                        "reset to avoid leaving a stuck GPU process behind."
                    ),
                ) from exc
            raise HTTPException(
                status_code=503,
                detail="The WSL vLLM runtime is unavailable on localhost.",
            ) from exc

        reply = "".join(reply_chunks).strip()
        thought = "".join(thought_chunks).strip() or None
        response_payload = {
            "reply": reply,
            "thought": thought,
            "raw_response": reply,
            "parsed": {"role": "assistant", "content": reply},
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "prompt_tokens": None,
            "generated_tokens": None,
            "active_model_key": spec["key"],
            "active_model": serialize_model_spec(spec),
            "active_quantization_key": quantization_spec["key"],
            "active_quantization": serialize_quantization_spec(quantization_spec),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_allocated_gib": 0.0,
            "vram_reserved_gib": 0.0,
        }
        if tts_enabled and reply:
            try:
                response_payload["tts_audio"] = tts_service.synthesize(reply)
            except Exception:
                logger.exception(
                    "TTS synthesis failed after WSL vLLM streaming key=%s quantization=%s",
                    spec["key"],
                    quantization_spec["key"],
                )
                response_payload["tts_audio"] = None
                response_payload["tts_audio_error"] = "Local TTS synthesis failed."
        yield {"event": "done", "payload": response_payload}


class GemmaService:
    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._current_model_key = None
        self._current_quantization_key = None
        self._runtime_family = None
        self._llama_cpp_runtime = LlamaCppServerRuntime()
        self._wsl_vllm_runtime = WslVllmServerRuntime()
        self._load_lock = threading.Lock()
        self._load_state_lock = threading.Lock()
        self._load_thread = None
        self._generate_lock = threading.Lock()
        self.loaded_at = None
        self._load_state = self._make_load_state(
            status="idle",
            progress=0,
            message="Select a model and click Load model.",
            target_model_key=DEFAULT_MODEL_KEY,
            target_quantization_key=DEFAULT_QUANTIZATION_KEY,
            error=None,
            started_at=None,
            finished_at=None,
        )

    @property
    def is_loaded(self) -> bool:
        if self._runtime_family == "llama.cpp":
            return self._llama_cpp_runtime.is_loaded
        if self._runtime_family == "vllm-wsl":
            return self._wsl_vllm_runtime.is_loaded
        return self._processor is not None and self._model is not None

    def _make_load_state(
        self,
        *,
        status: str,
        progress: int,
        message: str,
        target_model_key: str | None,
        target_quantization_key: str | None,
        error: str | None,
        started_at: float | None,
        finished_at: float | None,
    ) -> dict:
        target_spec = (
            serialize_model_spec(get_model_spec(target_model_key))
            if target_model_key is not None
            else None
        )
        target_quantization = (
            serialize_quantization_spec(get_quantization_spec(target_quantization_key))
            if target_quantization_key is not None
            else None
        )
        return {
            "status": status,
            "is_loading": status in {"queued", "loading"},
            "progress": max(0, min(100, int(progress))),
            "message": message,
            "error": error,
            "target_model_key": target_model_key,
            "target_model": target_spec,
            "target_quantization_key": target_quantization_key,
            "target_quantization": target_quantization,
            "started_at": started_at,
            "finished_at": finished_at,
            "updated_at": time.time(),
        }

    def _set_load_state(self, **updates: object) -> dict:
        with self._load_state_lock:
            current = dict(self._load_state)
            next_status = str(updates.get("status", current["status"]))
            target_model_key = updates.get(
                "target_model_key", current["target_model_key"]
            )
            target_quantization_key = updates.get(
                "target_quantization_key", current["target_quantization_key"]
            )
            started_at = updates.get("started_at", current["started_at"])
            finished_at = updates.get("finished_at", current["finished_at"])

            if next_status in {"queued", "loading"}:
                if started_at is None:
                    started_at = time.time()
                finished_at = None
            elif next_status == "idle":
                started_at = None
                finished_at = None
            elif next_status in {"loaded", "failed"} and "finished_at" not in updates:
                finished_at = time.time()

            self._load_state = self._make_load_state(
                status=next_status,
                progress=int(updates.get("progress", current["progress"])),
                message=str(updates.get("message", current["message"])),
                target_model_key=(
                    str(target_model_key) if target_model_key is not None else None
                ),
                target_quantization_key=(
                    str(target_quantization_key)
                    if target_quantization_key is not None
                    else None
                ),
                error=(
                    str(updates["error"])
                    if updates.get("error") is not None
                    else None
                ),
                started_at=started_at,
                finished_at=finished_at,
            )
            return dict(self._load_state)

    def get_load_state(self) -> dict:
        self._reconcile_runtime_state()
        with self._load_state_lock:
            return dict(self._load_state)

    def _reconcile_runtime_state(self) -> None:
        stale_runtime = None
        if self._runtime_family == "llama.cpp" and not self._llama_cpp_runtime.is_loaded:
            stale_runtime = "llama.cpp"
        elif self._runtime_family == "vllm-wsl" and not self._wsl_vllm_runtime.is_loaded:
            stale_runtime = "vllm-wsl"
        elif self._runtime_family == "transformers" and (
            self._processor is None or self._model is None
        ):
            stale_runtime = "transformers"

        if stale_runtime is None:
            return

        stale_model_key = self._current_model_key
        stale_quantization_key = self._current_quantization_key
        with self._load_state_lock:
            current_state = dict(self._load_state)

        self._unload_current_model()

        if (
            current_state.get("status") == "loaded"
            and stale_model_key is not None
            and stale_quantization_key is not None
        ):
            runtime_label = {
                "llama.cpp": "The llama.cpp runtime",
                "vllm-wsl": "The WSL vLLM runtime",
                "transformers": "The Transformers runtime",
            }.get(stale_runtime, "The model runtime")
            self._set_load_state(
                status="failed",
                progress=100,
                message="Model runtime stopped.",
                target_model_key=stale_model_key,
                target_quantization_key=stale_quantization_key,
                error=(
                    f"{runtime_label} exited after the load completed. Click Load model to "
                    "retry. If it keeps happening, free GPU memory or pick a lighter "
                    "quantization."
                ),
            )

    def _unload_current_model(self) -> None:
        self._llama_cpp_runtime.unload()
        self._wsl_vllm_runtime.unload()
        self._processor = None
        self._model = None
        self._current_model_key = None
        self._current_quantization_key = None
        self._runtime_family = None
        self.loaded_at = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_messages(
        self,
        *,
        prompt: str,
        system_prompt: str,
        history: list[HistoryTurn],
        image: Image.Image | None,
        audio: np.ndarray | None,
    ) -> list[dict]:
        messages: list[dict] = []
        if system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt.strip()}],
                }
            )

        for turn in history:
            messages.append(
                {
                    "role": turn.role,
                    "content": [{"type": "text", "text": turn.content}],
                }
            )

        current_content = []
        if image is not None:
            current_content.append({"type": "image", "image": image})
        if audio is not None:
            current_content.append({"type": "audio", "audio": audio})
        current_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": current_content})
        return messages

    def _build_generation_kwargs(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> dict:
        generation_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )
        else:
            generation_kwargs["do_sample"] = False
        return generation_kwargs

    def _serialize_generation_payload(
        self,
        *,
        processor: object,
        outputs: object,
        input_len: int,
        spec: dict,
        quantization_spec: dict,
        elapsed_ms: float,
    ) -> dict:
        generated_tokens = int(outputs[0].shape[-1] - input_len)
        raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed = processor.parse_response(raw_response)
        parsed_reply = extract_reply(parsed)
        return {
            "reply": parsed_reply["reply"],
            "thought": parsed_reply["thought"],
            "raw_response": raw_response,
            "parsed": parsed,
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": input_len,
            "generated_tokens": generated_tokens,
            "active_model_key": spec["key"],
            "active_model": serialize_model_spec(spec),
            "active_quantization_key": quantization_spec["key"],
            "active_quantization": serialize_quantization_spec(quantization_spec),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_allocated_gib": round(torch.cuda.memory_allocated() / (1024**3), 2)
            if torch.cuda.is_available()
            else 0.0,
            "vram_reserved_gib": round(torch.cuda.memory_reserved() / (1024**3), 2)
            if torch.cuda.is_available()
            else 0.0,
        }

    def ensure_loaded(
        self,
        model_key: str | None,
        quantization_key: str | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> tuple[object, object, dict, dict]:
        spec = get_model_spec(model_key)
        quantization_spec = get_quantization_spec(quantization_key)
        requested_key = spec["key"]
        requested_quantization_key = quantization_spec["key"]

        def emit(progress: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(progress, message)
        runtime_family = quantization_spec.get("runtime_family")

        if (
            self.is_loaded
            and self._runtime_family == runtime_family
            and self._current_model_key == requested_key
            and self._current_quantization_key == requested_quantization_key
        ):
            emit(
                100,
                f"{spec['label']} is already loaded in {quantization_spec['label']}.",
            )
            if runtime_family == "transformers":
                return self._processor, self._model, spec, quantization_spec
            return None, None, spec, quantization_spec

        with self._load_lock:
            if (
                self.is_loaded
                and self._runtime_family == runtime_family
                and self._current_model_key == requested_key
                and self._current_quantization_key == requested_quantization_key
            ):
                emit(
                    100,
                    f"{spec['label']} is already loaded in {quantization_spec['label']}.",
                )
                if runtime_family == "transformers":
                    return self._processor, self._model, spec, quantization_spec
                return None, None, spec, quantization_spec

            if self.is_loaded:
                emit(18, "Releasing the previous model from VRAM...")
                self._unload_current_model()

            emit(8, "Checking quantization support...")
            quantization_error = preflight_quantization_support(spec, quantization_spec)
            if quantization_error is not None:
                logger.warning(
                    "Model load blocked key=%s quantization=%s detail=%s",
                    spec["key"],
                    quantization_spec["key"],
                    quantization_error,
                )
                raise HTTPException(status_code=501, detail=quantization_error)

            if runtime_family == "llama.cpp":
                try:
                    self._llama_cpp_runtime.load(
                        spec,
                        quantization_spec,
                        progress_callback=emit,
                    )
                    self._current_model_key = requested_key
                    self._current_quantization_key = requested_quantization_key
                    self._runtime_family = "llama.cpp"
                    self.loaded_at = time.time()
                    logger.info(
                        "Loaded llama.cpp model key=%s quantization=%s hf_model_id=%s",
                        spec["key"],
                        quantization_spec["key"],
                        spec["hf_model_id"],
                    )
                    return None, None, spec, quantization_spec
                except HTTPException:
                    self._unload_current_model()
                    raise
                except Exception as exc:
                    self._unload_current_model()
                    detail = render_model_load_error(spec, exc)
                    logger.exception(
                        "llama.cpp model load failed key=%s quantization=%s hf_model_id=%s detail=%s",
                        spec["key"],
                        quantization_spec["key"],
                        spec["hf_model_id"],
                        detail,
                    )
                    raise HTTPException(status_code=503, detail=detail) from exc

            if runtime_family == "vllm-wsl":
                try:
                    self._wsl_vllm_runtime.load(
                        spec,
                        quantization_spec,
                        progress_callback=emit,
                    )
                    self._current_model_key = requested_key
                    self._current_quantization_key = requested_quantization_key
                    self._runtime_family = "vllm-wsl"
                    self.loaded_at = time.time()
                    logger.info(
                        "Loaded WSL vLLM model key=%s quantization=%s hf_model_id=%s",
                        spec["key"],
                        quantization_spec["key"],
                        spec["hf_model_id"],
                    )
                    return None, None, spec, quantization_spec
                except HTTPException:
                    self._unload_current_model()
                    raise
                except Exception as exc:
                    self._unload_current_model()
                    detail = render_model_load_error(spec, exc)
                    logger.exception(
                        "WSL vLLM model load failed key=%s quantization=%s hf_model_id=%s detail=%s",
                        spec["key"],
                        quantization_spec["key"],
                        spec["hf_model_id"],
                        detail,
                    )
                    raise HTTPException(status_code=503, detail=detail) from exc

            if runtime_family != "transformers":
                raise HTTPException(
                    status_code=501,
                    detail=(
                        f"{quantization_spec['label']} uses an unsupported runtime family: "
                        f"{runtime_family}"
                    ),
                )

            emit(16, "Checking workstation memory and commit availability...")
            preflight_error = preflight_model_load(spec)
            if preflight_error is not None:
                logger.warning(
                    "Model load blocked key=%s quantization=%s hf_model_id=%s detail=%s",
                    spec["key"],
                    quantization_spec["key"],
                    spec["hf_model_id"],
                    preflight_error,
                )
                raise HTTPException(status_code=503, detail=preflight_error)

            emit(24, "Preparing local cache and tokenizer assets...")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            try:
                emit(46, "Loading processor assets from the local cache...")
                self._processor = AutoProcessor.from_pretrained(
                    spec["hf_model_id"],
                    cache_dir=str(CACHE_DIR),
                    local_files_only=LOCAL_FILES_ONLY,
                )
                emit(78, "Loading model weights into VRAM...")
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    spec["hf_model_id"],
                    cache_dir=str(CACHE_DIR),
                    local_files_only=LOCAL_FILES_ONLY,
                    dtype=torch.bfloat16,
                    device_map="auto",
                )
                emit(94, "Finalizing the model session...")
                self._current_model_key = requested_key
                self._current_quantization_key = requested_quantization_key
                self._runtime_family = "transformers"
                self.loaded_at = time.time()
                emit(100, f"{spec['label']} is ready in {quantization_spec['label']}.")
                logger.info(
                    "Loaded model key=%s quantization=%s hf_model_id=%s",
                    spec["key"],
                    quantization_spec["key"],
                    spec["hf_model_id"],
                )
                return self._processor, self._model, spec, quantization_spec
            except Exception as exc:
                self._unload_current_model()
                detail = render_model_load_error(spec, exc)
                logger.exception(
                    "Model load failed key=%s quantization=%s hf_model_id=%s detail=%s",
                    spec["key"],
                    quantization_spec["key"],
                    spec["hf_model_id"],
                    detail,
                )
                raise HTTPException(status_code=503, detail=detail) from exc

    def _load_model_task(self, model_key: str, quantization_key: str) -> None:
        spec = get_model_spec(model_key)
        quantization_spec = get_quantization_spec(quantization_key)
        self._set_load_state(
            status="loading",
            progress=4,
            message=f"Starting {spec['label']} in {quantization_spec['label']}...",
            target_model_key=spec["key"],
            target_quantization_key=quantization_spec["key"],
            error=None,
        )

        try:
            self.ensure_loaded(
                spec["key"],
                quantization_spec["key"],
                progress_callback=lambda progress, message: self._set_load_state(
                    status="loaded" if progress >= 100 else "loading",
                    progress=progress,
                    message=message,
                    target_model_key=spec["key"],
                    target_quantization_key=quantization_spec["key"],
                    error=None,
                ),
            )
        except HTTPException as exc:
            current_progress = self.get_load_state().get("progress", 0)
            self._set_load_state(
                status="failed",
                progress=current_progress,
                message="Model load failed.",
                target_model_key=spec["key"],
                target_quantization_key=quantization_spec["key"],
                error=str(exc.detail),
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception(
                "Unexpected load task failure key=%s quantization=%s",
                spec["key"],
                quantization_spec["key"],
            )
            current_progress = self.get_load_state().get("progress", 0)
            self._set_load_state(
                status="failed",
                progress=current_progress,
                message="Model load failed.",
                target_model_key=spec["key"],
                target_quantization_key=quantization_spec["key"],
                error=render_model_load_error(spec, exc),
            )
        finally:
            with self._load_state_lock:
                self._load_thread = None

    def start_load(self, model_key: str | None, quantization_key: str | None) -> dict:
        self._reconcile_runtime_state()
        spec = get_model_spec(model_key)
        quantization_spec = get_quantization_spec(quantization_key)

        if (
            self.is_loaded
            and self._current_model_key == spec["key"]
            and self._current_quantization_key == quantization_spec["key"]
        ):
            state = self._set_load_state(
                status="loaded",
                progress=100,
                message=f"{spec['label']} is already loaded in {quantization_spec['label']}.",
                target_model_key=spec["key"],
                target_quantization_key=quantization_spec["key"],
                error=None,
            )
            return {
                "accepted": False,
                "message": state["message"],
                "load_state": state,
            }

        with self._load_state_lock:
            current = dict(self._load_state)
            if current["is_loading"]:
                same_target = (
                    current["target_model_key"] == spec["key"]
                    and current["target_quantization_key"] == quantization_spec["key"]
                )
                if same_target:
                    return {
                        "accepted": False,
                        "message": current["message"],
                        "load_state": current,
                    }

                active_target = current.get("target_model", {})
                active_quantization = current.get("target_quantization", {})
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{active_target.get('label', 'Another model')} / "
                        f"{active_quantization.get('label', 'current quantization')} is "
                        "already loading. Wait for it to finish before starting another load."
                    ),
                )

            state = self._make_load_state(
                status="queued",
                progress=2,
                message=f"Queued {spec['label']} in {quantization_spec['label']}.",
                target_model_key=spec["key"],
                target_quantization_key=quantization_spec["key"],
                error=None,
                started_at=time.time(),
                finished_at=None,
            )
            self._load_state = state
            self._load_thread = threading.Thread(
                target=self._load_model_task,
                args=(spec["key"], quantization_spec["key"]),
                daemon=True,
                name=f"gemma-load-{spec['key']}-{quantization_spec['key']}",
            )
            load_thread = self._load_thread

        logger.info(
            "Model load queued key=%s quantization=%s",
            spec["key"],
            quantization_spec["key"],
        )
        load_thread.start()
        return {
            "accepted": True,
            "message": state["message"],
            "load_state": state,
        }

    def require_loaded_selection(
        self, model_key: str | None, quantization_key: str | None
    ) -> tuple[object, object, dict, dict]:
        self._reconcile_runtime_state()
        spec = get_model_spec(model_key)
        quantization_spec = get_quantization_spec(quantization_key)
        load_state = self.get_load_state()

        if load_state["is_loading"]:
            target_model = load_state.get("target_model")
            target_quantization = load_state.get("target_quantization")
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{target_model['label']} / {target_quantization['label']} is still "
                    "loading. Wait for the progress bar to finish before sending a turn."
                ),
            )

        if not self.is_loaded:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"No model is loaded yet. Click Load model for {spec['label']} in "
                    f"{quantization_spec['label']} first."
                ),
            )

        if (
            self._current_model_key != spec["key"]
            or self._current_quantization_key != quantization_spec["key"]
        ):
            loaded_spec = get_model_spec(self._current_model_key)
            loaded_quantization = get_quantization_spec(self._current_quantization_key)
            raise HTTPException(
                status_code=409,
                detail=(
                    f"VRAM currently holds {loaded_spec['label']} in "
                    f"{loaded_quantization['label']}. Click Load model to switch to "
                    f"{spec['label']} / {quantization_spec['label']} before sending."
                ),
            )

        return self._processor, self._model, spec, quantization_spec

    def health(self) -> dict:
        self._reconcile_runtime_state()
        active_spec = None
        active_quantization = None
        if self.is_loaded:
            active_spec = (
                get_model_spec(self._current_model_key)
                if self._current_model_key is not None
                else None
            )
            active_quantization = (
                get_quantization_spec(self._current_quantization_key)
                if self._current_quantization_key is not None
                else None
            )
        windows_commit_snapshot = get_windows_commit_snapshot()
        gpu_total_memory_gib = get_gpu_total_memory_gib()
        return {
            "status": "ready",
            "active_model_key": active_spec["key"] if active_spec else None,
            "active_model": serialize_model_spec(active_spec) if active_spec else None,
            "active_quantization_key": (
                active_quantization["key"] if active_quantization else None
            ),
            "active_quantization": (
                serialize_quantization_spec(active_quantization)
                if active_quantization
                else None
            ),
            "runtime_family": self._runtime_family,
            "tts": tts_service.health(),
            "loaded": self.is_loaded,
            "local_files_only": LOCAL_FILES_ONLY,
            "cache_dir": str(CACHE_DIR),
            "log_path": str(LOG_PATH),
            "wsl_vllm_log_path": str(WSL_VLLM_LOG_PATH),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "gpu_total_memory_gib": gpu_total_memory_gib,
            "vram_allocated_gib": round(torch.cuda.memory_allocated() / (1024**3), 2)
            if torch.cuda.is_available() and self.is_loaded
            else 0.0,
            "vram_reserved_gib": round(torch.cuda.memory_reserved() / (1024**3), 2)
            if torch.cuda.is_available() and self.is_loaded
            else 0.0,
            "windows_commit_available_gib": windows_commit_snapshot["available_commit_gib"]
            if windows_commit_snapshot
            else None,
            "windows_commit_limit_gib": windows_commit_snapshot["commit_limit_gib"]
            if windows_commit_snapshot
            else None,
            "loaded_at": self.loaded_at,
            "load_state": self.get_load_state(),
        }

    def generate(
        self,
        *,
        model_key: str,
        quantization_key: str,
        prompt: str,
        system_prompt: str,
        history: list[HistoryTurn],
        image: Image.Image | None,
        audio: np.ndarray | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        thinking: bool,
    ) -> dict:
        processor, model, spec, quantization_spec = self.require_loaded_selection(
            model_key, quantization_key
        )
        if self._runtime_family == "llama.cpp":
            if image is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{spec['label']} in {quantization_spec['label']} is currently wired "
                        "for fast text chat only in this app build."
                    ),
                )
            if audio is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{spec['label']} in {quantization_spec['label']} does not support "
                        "audio input in this runtime path."
                    ),
                )
            return self._llama_cpp_runtime.generate(
                spec=spec,
                quantization_spec=quantization_spec,
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        if self._runtime_family == "vllm-wsl":
            if audio is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{spec['label']} in {quantization_spec['label']} does not support "
                        "audio input in this runtime path."
                    ),
                )
            return self._wsl_vllm_runtime.generate(
                spec=spec,
                quantization_spec=quantization_spec,
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
                image=image,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            image=image,
            audio=audio,
        )

        start_time = time.perf_counter()
        with self._generate_lock:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=thinking,
            ).to(model.device)
            input_len = int(inputs["input_ids"].shape[-1])
            generation_kwargs = self._build_generation_kwargs(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_kwargs)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        return self._serialize_generation_payload(
            processor=processor,
            outputs=outputs,
            input_len=input_len,
            spec=spec,
            quantization_spec=quantization_spec,
            elapsed_ms=elapsed_ms,
        )

    def stream_generate(
        self,
        *,
        model_key: str,
        quantization_key: str,
        prompt: str,
        system_prompt: str,
        history: list[HistoryTurn],
        image: Image.Image | None,
        audio: np.ndarray | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        thinking: bool,
        tts_enabled: bool,
    ):
        processor, model, spec, quantization_spec = self.require_loaded_selection(
            model_key, quantization_key
        )
        if self._runtime_family == "llama.cpp":
            if image is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{spec['label']} in {quantization_spec['label']} is currently wired "
                        "for fast text chat only in this app build."
                    ),
                )
            if audio is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{spec['label']} in {quantization_spec['label']} does not support "
                        "audio input in this runtime path."
                    ),
                )
            yield from self._llama_cpp_runtime.stream_generate(
                spec=spec,
                quantization_spec=quantization_spec,
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                tts_enabled=tts_enabled,
            )
            return
        if self._runtime_family == "vllm-wsl":
            if audio is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{spec['label']} in {quantization_spec['label']} does not support "
                        "audio input in this runtime path."
                    ),
                )
            yield from self._wsl_vllm_runtime.stream_generate(
                spec=spec,
                quantization_spec=quantization_spec,
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
                image=image,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                tts_enabled=tts_enabled,
            )
            return
        messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            image=image,
            audio=audio,
        )

        start_time = time.perf_counter()
        with self._generate_lock:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=thinking,
            ).to(model.device)
            input_len = int(inputs["input_ids"].shape[-1])
            generation_kwargs = self._build_generation_kwargs(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            streamer = TextIteratorStreamer(
                processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=False,
            )
            generation_kwargs["streamer"] = streamer

            result_holder: dict[str, object] = {}
            error_holder: dict[str, Exception] = {}

            def run_generation() -> None:
                try:
                    with torch.inference_mode():
                        result_holder["outputs"] = model.generate(**inputs, **generation_kwargs)
                except Exception as exc:  # pragma: no cover - runtime safety
                    error_holder["error"] = exc
                    logger.exception(
                        "Streaming generation failed key=%s quantization=%s",
                        spec["key"],
                        quantization_spec["key"],
                    )
                finally:
                    streamer.on_finalized_text("", stream_end=True)

            generation_thread = threading.Thread(
                target=run_generation,
                daemon=True,
                name=f"gemma-stream-{spec['key']}-{quantization_spec['key']}",
            )
            generation_thread.start()

            yield {"event": "start"}
            for text_chunk in streamer:
                if text_chunk:
                    yield {"event": "token", "text": text_chunk}

            generation_thread.join()

        if "error" in error_holder:
            raise error_holder["error"]

        outputs = result_holder.get("outputs")
        if outputs is None:
            raise RuntimeError("Streaming generation finished without output tokens.")

        response_payload = self._serialize_generation_payload(
            processor=processor,
            outputs=outputs,
            input_len=input_len,
            spec=spec,
            quantization_spec=quantization_spec,
            elapsed_ms=round((time.perf_counter() - start_time) * 1000, 1),
        )

        if tts_enabled and response_payload.get("reply"):
            try:
                response_payload["tts_audio"] = tts_service.synthesize(response_payload["reply"])
            except Exception:
                logger.exception(
                    "TTS synthesis failed after streaming key=%s quantization=%s",
                    spec["key"],
                    quantization_spec["key"],
                )
                response_payload["tts_audio"] = None
                response_payload["tts_audio_error"] = "Local TTS synthesis failed."

        yield {"event": "done", "payload": response_payload}


def extract_reply(parsed: object) -> dict:
    if isinstance(parsed, dict):
        thought = parsed.get("thought")
        answer = parsed.get("answer")
        content = parsed.get("content")
        return {
            "reply": answer or content or str(parsed),
            "thought": thought,
        }

    return {"reply": str(parsed), "thought": None}


def decode_history(history_json: str) -> list[HistoryTurn]:
    if not history_json.strip():
        return []

    try:
        raw_history = json.loads(history_json)
        return [HistoryTurn.model_validate(item) for item in raw_history]
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid history payload: {exc}") from exc


def load_image(upload: UploadFile | None) -> Image.Image | None:
    if upload is None:
        return None

    try:
        data = upload.file.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unsupported image file.") from exc


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)

    new_length = max(1, int(math.ceil(len(audio) * target_rate / source_rate)))
    old_positions = np.arange(len(audio), dtype=np.float32)
    new_positions = np.linspace(
        0,
        max(len(audio) - 1, 1),
        new_length,
        dtype=np.float32,
    )
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def load_audio(upload: UploadFile | None, target_rate: int) -> np.ndarray | None:
    if upload is None:
        return None

    try:
        data = upload.file.read()
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail="Unsupported audio file.") from exc

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    return resample_audio(audio, sample_rate, target_rate)


def prepare_generation_request(
    *,
    model_key: str,
    quantization_key: str,
    prompt: str,
    system_prompt: str,
    history_json: str,
    image: UploadFile | None,
    audio: UploadFile | None,
) -> tuple[dict, dict, str, str, list[HistoryTurn], Image.Image | None, np.ndarray | None]:
    spec = get_model_spec(model_key)
    quantization_spec = get_quantization_spec(quantization_key)
    logger.info(
        "Generate requested key=%s quantization=%s image=%s audio=%s",
        spec["key"],
        quantization_spec["key"],
        image is not None,
        audio is not None,
    )
    history = decode_history(history_json)

    if quantization_spec.get("runtime_family") == "llama.cpp" and image is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{spec['label']} in {quantization_spec['label']} is currently wired for "
                "fast text chat only in this app build."
            ),
        )

    if quantization_spec.get("runtime_family") == "llama.cpp" and audio is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{spec['label']} in {quantization_spec['label']} does not support audio "
                "input in this runtime path."
            ),
        )

    image_payload = load_image(image)

    if image_payload is not None and not spec["supports_image"]:
        raise HTTPException(
            status_code=400,
            detail=f"{spec['label']} does not support image input in the official tables.",
        )

    if audio is not None and not spec["supports_audio"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{spec['label']} does not support audio input according to the official "
                "Gemma 4 Supported Modalities tables."
            ),
        )

    processor, _, _, _ = service.require_loaded_selection(
        spec["key"], quantization_spec["key"]
    )
    target_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    audio_payload = load_audio(audio, target_rate)

    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        if image_payload is not None and audio_payload is not None:
            normalized_prompt = "Describe the image and transcribe the audio."
        elif image_payload is not None:
            normalized_prompt = "Describe this image."
        elif audio_payload is not None:
            normalized_prompt = "Transcribe this audio."
        else:
            raise HTTPException(status_code=400, detail="Prompt, image, or audio is required.")

    return (
        spec,
        quantization_spec,
        normalized_prompt,
        system_prompt or DEFAULT_SYSTEM_PROMPT,
        history[-8:],
        image_payload,
        audio_payload,
    )


service = GemmaService()
tts_service = LocalTTSService()
app = FastAPI(title="Gemma 4 Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def disable_static_caching(request: Request, call_next):
    response = await call_next(request)
    if request.method == "GET" and not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.on_event("startup")
def on_startup() -> None:
    logger.info(
        "Gemma 4 Lab started cache_dir=%s local_files_only=%s tts_voice=%s",
        CACHE_DIR,
        LOCAL_FILES_ONLY,
        DEFAULT_TTS_VOICE,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception path=%s method=%s", request.url.path, request.method)
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "Internal server error. See logs/gemma4-lab.log for the full traceback."
            )
        },
    )


@app.get("/api/health")
def api_health() -> dict:
    return service.health()


@app.get("/api/models")
def api_models() -> dict:
    health = service.health()
    return {
        "default_model_key": DEFAULT_MODEL_KEY,
        "default_quantization_key": DEFAULT_QUANTIZATION_KEY,
        "active_model_key": health["active_model_key"],
        "active_quantization_key": health["active_quantization_key"],
        "docs_note": DOCS_NOTE,
        "quantization_note": QUANTIZATION_NOTE,
        "models": [serialize_model_spec(spec) for spec in MODEL_SPECS],
        "quantizations": [serialize_quantization_spec(spec) for spec in QUANTIZATION_SPECS],
    }


@app.get("/api/models/load-status")
def api_model_load_status() -> dict:
    return {
        "load_state": service.get_load_state(),
    }


@app.post("/api/models/load")
def api_load_model(request: ModelLoadRequest) -> dict:
    logger.info(
        "Model load requested key=%s quantization=%s",
        request.model_key,
        request.quantization_key,
    )
    load_result = service.start_load(request.model_key, request.quantization_key)
    return {
        "accepted": load_result["accepted"],
        "message": load_result["message"],
        "load_state": load_result["load_state"],
        "health": service.health(),
    }


@app.post("/api/generate")
async def api_generate(
    model_key: str = Form(DEFAULT_MODEL_KEY),
    quantization_key: str = Form(DEFAULT_QUANTIZATION_KEY),
    prompt: str = Form(""),
    system_prompt: str = Form(DEFAULT_SYSTEM_PROMPT),
    history_json: str = Form("[]"),
    max_new_tokens: int = Form(256),
    temperature: float = Form(1.0),
    top_p: float = Form(0.95),
    top_k: int = Form(64),
    thinking: bool = Form(False),
    tts_enabled: bool = Form(False),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
) -> dict:
    spec, quantization_spec, normalized_prompt, normalized_system_prompt, history, image_payload, audio_payload = (
        prepare_generation_request(
            model_key=model_key,
            quantization_key=quantization_key,
            prompt=prompt,
            system_prompt=system_prompt,
            history_json=history_json,
            image=image,
            audio=audio,
        )
    )
    logger.info(
        "Generate execution key=%s quantization=%s thinking=%s tts=%s",
        spec["key"],
        quantization_spec["key"],
        thinking,
        tts_enabled,
    )

    response_payload = service.generate(
        model_key=spec["key"],
        quantization_key=quantization_spec["key"],
        prompt=normalized_prompt,
        system_prompt=normalized_system_prompt,
        history=history,
        image=image_payload,
        audio=audio_payload,
        max_new_tokens=max(32, min(max_new_tokens, 1024)),
        temperature=max(0.0, min(temperature, 2.0)),
        top_p=max(0.1, min(top_p, 1.0)),
        top_k=max(1, min(top_k, 128)),
        thinking=thinking,
    )

    if tts_enabled and response_payload.get("reply"):
        try:
            response_payload["tts_audio"] = tts_service.synthesize(response_payload["reply"])
        except Exception:
            logger.exception(
                "TTS synthesis failed key=%s quantization=%s",
                spec["key"],
                quantization_spec["key"],
            )
            response_payload["tts_audio"] = None
            response_payload["tts_audio_error"] = "Local TTS synthesis failed."

    return response_payload


@app.post("/api/generate-stream")
async def api_generate_stream(
    model_key: str = Form(DEFAULT_MODEL_KEY),
    quantization_key: str = Form(DEFAULT_QUANTIZATION_KEY),
    prompt: str = Form(""),
    system_prompt: str = Form(DEFAULT_SYSTEM_PROMPT),
    history_json: str = Form("[]"),
    max_new_tokens: int = Form(256),
    temperature: float = Form(1.0),
    top_p: float = Form(0.95),
    top_k: int = Form(64),
    thinking: bool = Form(False),
    tts_enabled: bool = Form(False),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
) -> StreamingResponse:
    spec, quantization_spec, normalized_prompt, normalized_system_prompt, history, image_payload, audio_payload = (
        prepare_generation_request(
            model_key=model_key,
            quantization_key=quantization_key,
            prompt=prompt,
            system_prompt=system_prompt,
            history_json=history_json,
            image=image,
            audio=audio,
        )
    )
    logger.info(
        "Generate stream execution key=%s quantization=%s thinking=%s tts=%s",
        spec["key"],
        quantization_spec["key"],
        thinking,
        tts_enabled,
    )

    def event_stream():
        try:
            for event in service.stream_generate(
                model_key=spec["key"],
                quantization_key=quantization_spec["key"],
                prompt=normalized_prompt,
                system_prompt=normalized_system_prompt,
                history=history,
                image=image_payload,
                audio=audio_payload,
                max_new_tokens=max(32, min(max_new_tokens, 1024)),
                temperature=max(0.0, min(temperature, 2.0)),
                top_p=max(0.1, min(top_p, 1.0)),
                top_k=max(1, min(top_k, 128)),
                thinking=thinking,
                tts_enabled=tts_enabled,
            ):
                yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        except HTTPException as exc:
            yield (
                json.dumps(
                    {"event": "error", "detail": str(exc.detail)},
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception(
                "Unhandled stream error key=%s quantization=%s",
                spec["key"],
                quantization_spec["key"],
            )
            yield (
                json.dumps(
                    {"event": "error", "detail": str(render_model_load_error(spec, exc))},
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/api/tts/clips/{clip_id}.wav")
def api_tts_clip(clip_id: str) -> FileResponse:
    if not CLIP_ID_PATTERN.fullmatch(clip_id):
        raise HTTPException(status_code=404, detail="TTS clip not found.")

    clip_path = TTS_GENERATED_DIR / f"{clip_id}.wav"
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="TTS clip not found.")

    return FileResponse(clip_path, media_type="audio/wav", filename=clip_path.name)


if DIST_DIR.exists():

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(DIST_DIR / "index.html")

    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="web")
