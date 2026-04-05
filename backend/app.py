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
import sqlite3
import subprocess
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
import wave
import socket
from queue import Queue
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Literal

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
DATA_DIR = BASE_DIR / "data"
LOG_PATH = LOG_DIR / "gemma4-lab.log"
LLAMA_SERVER_LOG_PATH = LOG_DIR / "llama-server.log"
WSL_VLLM_LOG_PATH = LOG_DIR / "vllm-wsl.log"
REQUESTS_DB_PATH = DATA_DIR / "gemma4-requests.sqlite3"
TTS_DIR = BASE_DIR / "tts"
TTS_VOICE_DIR = TTS_DIR / "voices"
TTS_GENERATED_DIR = TTS_DIR / "generated"
LLAMA_SERVER_BIN = BASE_DIR / "tools" / "llama.cpp" / "bin" / "llama-server.exe"
LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8011
LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
LLAMA_SERVER_CACHE_RAM_MIB = int(os.getenv("GEMMA4_LLAMA_CACHE_RAM_MIB", "2048"))
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
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("GEMMA4_MAX_NEW_TOKENS", "512"))
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
DATA_DIR.mkdir(parents=True, exist_ok=True)
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
        "min_windows_physical_available_gib": 50.0,
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


class ApiMediaUrl(BaseModel):
    url: str


class ApiMessageContentPart(BaseModel):
    type: Literal["text", "image_url", "audio_url"]
    text: str | None = None
    image_url: ApiMediaUrl | None = None
    audio_url: ApiMediaUrl | None = None


class ApiChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ApiMessageContentPart]
    name: str | None = None
    tool_call_id: str | None = None


class ApiResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"
    json_schema: dict[str, Any] | None = None


class ApiToolFunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ApiToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: ApiToolFunctionDefinition


class ApiToolChoiceFunction(BaseModel):
    name: str


class ApiToolChoiceObject(BaseModel):
    type: Literal["function"]
    function: ApiToolChoiceFunction


class ApiChatCompletionRequest(BaseModel):
    request_id: str | None = None
    model: str | None = None
    model_key: str | None = None
    quantization_key: str | None = None
    messages: list[ApiChatMessage]
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    stream: bool = False
    thinking: bool = False
    tts_enabled: bool = False
    response_format: ApiResponseFormat | None = None
    tools: list[ApiToolDefinition] | None = None
    tool_choice: str | ApiToolChoiceObject | None = "auto"


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


def build_runtime_capabilities(spec: dict | None, quantization_spec: dict | None) -> dict | None:
    if spec is None or quantization_spec is None:
        return None

    runtime_family = str(quantization_spec.get("runtime_family") or "")
    supports_text = bool(spec.get("supports_text", True))
    supports_image = bool(spec.get("supports_image")) and runtime_family in {
        "transformers",
        "llama.cpp",
        "vllm-wsl",
    }
    supports_audio = bool(spec.get("supports_audio")) and runtime_family == "transformers"
    supported_modalities: list[str] = []
    if supports_text:
        supported_modalities.append("text")
    if supports_image:
        supported_modalities.append("image")
    if supports_audio:
        supported_modalities.append("audio")

    return {
        "runtime_family": runtime_family,
        "supported_modalities": supported_modalities,
        "supports_text": supports_text,
        "supports_image": supports_image,
        "supports_audio": supports_audio,
    }


def get_gpu_total_memory_gib() -> float | None:
    if not torch.cuda.is_available():
        return None

    return round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)


def get_gpu_monitor_snapshot() -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        return None

    fallback = {
        "name": torch.cuda.get_device_name(0),
        "utilization_gpu_percent": None,
        "utilization_memory_percent": None,
        "memory_used_gib": round(torch.cuda.memory_reserved() / (1024**3), 2),
        "memory_total_gib": get_gpu_total_memory_gib(),
        "temperature_c": None,
        "power_draw_watts": None,
        "source": "torch",
    }

    try:
        query = ",".join(
            [
                "name",
                "utilization.gpu",
                "utilization.memory",
                "memory.used",
                "memory.total",
                "temperature.gpu",
                "power.draw",
            ]
        )
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return fallback

        first_line = next(
            (line.strip() for line in result.stdout.splitlines() if line.strip()),
            "",
        )
        if not first_line:
            return fallback

        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) < 7:
            return fallback

        def parse_number(value: str) -> float | None:
            normalized = value.strip().lower()
            if not normalized or normalized in {"n/a", "[n/a]"}:
                return None
            try:
                return float(value)
            except ValueError:
                return None

        memory_used_mib = parse_number(parts[3])
        memory_total_mib = parse_number(parts[4])
        return {
            "name": parts[0] or fallback["name"],
            "utilization_gpu_percent": parse_number(parts[1]),
            "utilization_memory_percent": parse_number(parts[2]),
            "memory_used_gib": round((memory_used_mib or 0.0) / 1024, 2)
            if memory_used_mib is not None
            else fallback["memory_used_gib"],
            "memory_total_gib": round((memory_total_mib or 0.0) / 1024, 2)
            if memory_total_mib is not None
            else fallback["memory_total_gib"],
            "temperature_c": parse_number(parts[5]),
            "power_draw_watts": parse_number(parts[6]),
            "source": "nvidia-smi",
        }
    except Exception:
        logger.exception("Failed to collect GPU monitoring snapshot")
        return fallback


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


def log_runtime_memory_snapshot(context: str) -> None:
    snapshot = get_windows_commit_snapshot()
    gpu_total = get_gpu_total_memory_gib()
    if snapshot is None:
        logger.info(
            "Memory snapshot context=%s gpu_total_gib=%s vram_allocated_gib=%s vram_reserved_gib=%s",
            context,
            gpu_total,
            round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else None,
            round(torch.cuda.memory_reserved() / (1024**3), 2) if torch.cuda.is_available() else None,
        )
        return

    logger.info(
        "Memory snapshot context=%s free_ram_gib=%s free_commit_gib=%s total_ram_gib=%s commit_limit_gib=%s gpu_total_gib=%s vram_allocated_gib=%s vram_reserved_gib=%s",
        context,
        snapshot["available_physical_gib"],
        snapshot["available_commit_gib"],
        snapshot["total_physical_gib"],
        snapshot["commit_limit_gib"],
        gpu_total,
        round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else None,
        round(torch.cuda.memory_reserved() / (1024**3), 2) if torch.cuda.is_available() else None,
    )


def preflight_model_load(spec: dict) -> str | None:
    snapshot = get_windows_commit_snapshot()
    if snapshot is None:
        return None

    required_physical_gib = spec.get("min_windows_physical_available_gib")
    if (
        required_physical_gib is not None
        and snapshot["available_physical_gib"] < required_physical_gib
    ):
        return (
            f"{spec['label']} was blocked before loading because Windows only has "
            f"{snapshot['available_physical_gib']:.2f} GiB of free RAM right now, and "
            f"this bf16 path needs about {spec['memory_requirements_gib'].get('bf16', required_physical_gib):.1f} GiB "
            "for the weights plus a bit of headroom for the Python runtime. Free some system "
            "memory and retry, or switch to SFP8 / Q4_0 for this model."
        )

    required_commit_gib = spec.get("min_windows_commit_available_gib")
    if (
        required_commit_gib is not None
        and snapshot["available_commit_gib"] < required_commit_gib
    ):
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


def infer_transformers_finish_reason(
    processor: object, outputs: object, input_len: int, requested_max_new_tokens: int
) -> tuple[str, bool]:
    generated_ids = outputs[0][input_len:].tolist()
    generated_tokens = len(generated_ids)

    eos_token_ids: set[int] = set()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            eos_token_ids.add(eos_token_id)
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_token_ids.update(
                token_id for token_id in eos_token_id if isinstance(token_id, int)
            )

    if generated_ids and eos_token_ids and generated_ids[-1] in eos_token_ids:
        return "stop", False

    if generated_tokens >= max(1, int(requested_max_new_tokens)):
        return "length", True

    return "stop", False


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

    stdout_text = result.stdout if isinstance(result.stdout, str) else ""
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


def get_llama_server_process_ids() -> set[int]:
    script = """
$items = Get-CimInstance Win32_Process -Filter "name = 'llama-server.exe'" |
  Select-Object ProcessId, CommandLine
$items | ConvertTo-Json -Compress
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception:
        logger.exception("Failed to inspect llama-server processes")
        return set()

    stdout_text = result.stdout if isinstance(result.stdout, str) else ""
    if not stdout_text.strip():
        return set()

    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError:
        logger.warning("Unable to parse llama-server process snapshot: %s", stdout_text[:400])
        return set()

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return set()

    runtime_bin = str(LLAMA_SERVER_BIN).lower()
    runtime_port = f"--port {LLAMA_SERVER_PORT}"
    pids: set[int] = set()

    for item in payload:
        if not isinstance(item, dict):
            continue
        command_line = str(item.get("CommandLine") or "")
        if not command_line:
            continue
        lowered = command_line.lower()
        if runtime_bin not in lowered and runtime_port not in lowered:
            continue
        process_id = item.get("ProcessId")
        if isinstance(process_id, int):
            pids.add(process_id)
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


class InferenceRequestStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._db_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_requests (
                    request_id TEXT PRIMARY KEY,
                    route TEXT NOT NULL,
                    status TEXT NOT NULL,
                    runtime_family TEXT,
                    model_key TEXT,
                    quantization_key TEXT,
                    stream INTEGER NOT NULL DEFAULT 0,
                    queue_position INTEGER,
                    progress_message TEXT,
                    request_payload_json TEXT,
                    response_payload_json TEXT,
                    response_preview TEXT,
                    error_text TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_inference_requests_created_at
                ON inference_requests(created_at DESC)
                """
            )
            connection.commit()

    def create_request(
        self,
        *,
        request_id: str,
        route: str,
        runtime_family: str | None,
        model_key: str | None,
        quantization_key: str | None,
        stream: bool,
        request_payload: dict[str, Any],
        queue_position: int,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO inference_requests (
                    request_id,
                    route,
                    status,
                    runtime_family,
                    model_key,
                    quantization_key,
                    stream,
                    queue_position,
                    progress_message,
                    request_payload_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    route,
                    "queued",
                    runtime_family,
                    model_key,
                    quantization_key,
                    int(stream),
                    queue_position,
                    "Queued for execution.",
                    json.dumps(request_payload, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            connection.commit()

    def update_queue_positions(
        self,
        *,
        active_request_id: str | None,
        queued_request_ids: list[str],
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as connection:
            if active_request_id is not None:
                connection.execute(
                    """
                    UPDATE inference_requests
                    SET queue_position = 0, updated_at = ?
                    WHERE request_id = ? AND status = 'running'
                    """,
                    (now, active_request_id),
                )
            for index, request_id in enumerate(queued_request_ids, start=1):
                connection.execute(
                    """
                    UPDATE inference_requests
                    SET queue_position = ?, progress_message = ?, updated_at = ?
                    WHERE request_id = ? AND status = 'queued'
                    """,
                    (index, f"Queued at position {index}.", now, request_id),
                )
            connection.commit()

    def mark_running(self, request_id: str, *, message: str) -> None:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE inference_requests
                SET status = 'running',
                    queue_position = 0,
                    progress_message = ?,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE request_id = ?
                """,
                (message, now, now, request_id),
            )
            connection.commit()

    def update_progress(
        self,
        request_id: str,
        *,
        status: str | None = None,
        message: str | None = None,
        response_preview: str | None = None,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as connection:
            current = self.get_request(request_id)
            if current is None:
                return
            connection.execute(
                """
                UPDATE inference_requests
                SET status = ?,
                    progress_message = ?,
                    response_preview = ?,
                    updated_at = ?
                WHERE request_id = ?
                """,
                (
                    status or current["status"],
                    message if message is not None else current["progress_message"],
                    response_preview
                    if response_preview is not None
                    else current["response_preview"],
                    now,
                    request_id,
                ),
            )
            connection.commit()

    def mark_completed(
        self,
        request_id: str,
        *,
        response_payload: dict[str, Any],
        response_preview: str | None,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE inference_requests
                SET status = 'completed',
                    queue_position = NULL,
                    progress_message = 'Completed.',
                    response_payload_json = ?,
                    response_preview = ?,
                    finished_at = ?,
                    updated_at = ?
                WHERE request_id = ?
                """,
                (
                    json.dumps(response_payload, ensure_ascii=False),
                    response_preview,
                    now,
                    now,
                    request_id,
                ),
            )
            connection.commit()

    def mark_failed(self, request_id: str, *, error_text: str) -> None:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE inference_requests
                SET status = 'failed',
                    queue_position = NULL,
                    progress_message = 'Failed.',
                    error_text = ?,
                    finished_at = ?,
                    updated_at = ?
                WHERE request_id = ?
                """,
                (error_text, now, now, request_id),
            )
            connection.commit()

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM inference_requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return self._serialize_row(row)

    def list_requests(self, *, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM inference_requests
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._serialize_row(row) for row in rows]

    def _serialize_row(self, row: sqlite3.Row) -> dict[str, Any]:
        request_payload = None
        response_payload = None
        if row["request_payload_json"]:
            try:
                request_payload = json.loads(row["request_payload_json"])
            except json.JSONDecodeError:
                request_payload = None
        if row["response_payload_json"]:
            try:
                response_payload = json.loads(row["response_payload_json"])
            except json.JSONDecodeError:
                response_payload = None

        return {
            "request_id": row["request_id"],
            "route": row["route"],
            "status": row["status"],
            "runtime_family": row["runtime_family"],
            "model_key": row["model_key"],
            "quantization_key": row["quantization_key"],
            "stream": bool(row["stream"]),
            "queue_position": row["queue_position"],
            "progress_message": row["progress_message"],
            "request_payload": request_payload,
            "response_payload": response_payload,
            "response_preview": row["response_preview"],
            "error_text": row["error_text"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "elapsed_ms": (
                round((row["finished_at"] - row["started_at"]) * 1000, 1)
                if row["finished_at"] is not None and row["started_at"] is not None
                else None
            ),
        }


class InferenceQueueManager:
    def __init__(self, request_store: InferenceRequestStore) -> None:
        self._request_store = request_store
        self._condition = threading.Condition()
        self._queued_request_ids: list[str] = []
        self._active_request_id: str | None = None

    def register_request(
        self,
        *,
        route: str,
        runtime_family: str | None,
        model_key: str | None,
        quantization_key: str | None,
        stream: bool,
        request_payload: dict[str, Any],
        request_id: str | None = None,
    ) -> str:
        resolved_request_id = (request_id or uuid.uuid4().hex).strip()
        with self._condition:
            self._queued_request_ids.append(resolved_request_id)
            queue_position = len(self._queued_request_ids)
            self._request_store.create_request(
                request_id=resolved_request_id,
                route=route,
                runtime_family=runtime_family,
                model_key=model_key,
                quantization_key=quantization_key,
                stream=stream,
                request_payload=request_payload,
                queue_position=queue_position,
            )
            self._request_store.update_queue_positions(
                active_request_id=self._active_request_id,
                queued_request_ids=list(self._queued_request_ids),
            )
            logger.info(
                "Inference request queued request_id=%s route=%s model=%s quantization=%s queue_position=%s",
                resolved_request_id,
                route,
                model_key,
                quantization_key,
                queue_position,
            )
        return resolved_request_id

    def wait_for_turn(self, request_id: str, *, message: str) -> None:
        with self._condition:
            while True:
                is_first = bool(self._queued_request_ids) and self._queued_request_ids[0] == request_id
                if self._active_request_id is None and is_first:
                    self._active_request_id = request_id
                    self._queued_request_ids.pop(0)
                    self._request_store.mark_running(request_id, message=message)
                    self._request_store.update_queue_positions(
                        active_request_id=self._active_request_id,
                        queued_request_ids=list(self._queued_request_ids),
                    )
                    logger.info("Inference request started request_id=%s", request_id)
                    return
                self._request_store.update_queue_positions(
                    active_request_id=self._active_request_id,
                    queued_request_ids=list(self._queued_request_ids),
                )
                self._condition.wait(timeout=0.25)

    def finish(
        self,
        request_id: str,
        *,
        response_payload: dict[str, Any] | None = None,
        response_preview: str | None = None,
        error_text: str | None = None,
    ) -> None:
        with self._condition:
            if request_id in self._queued_request_ids:
                self._queued_request_ids = [
                    queued_request_id
                    for queued_request_id in self._queued_request_ids
                    if queued_request_id != request_id
                ]
            if self._active_request_id == request_id:
                self._active_request_id = None

            if error_text is not None:
                self._request_store.mark_failed(request_id, error_text=error_text)
                logger.warning(
                    "Inference request failed request_id=%s error=%s",
                    request_id,
                    error_text,
                )
            elif response_payload is not None:
                self._request_store.mark_completed(
                    request_id,
                    response_payload=response_payload,
                    response_preview=response_preview,
                )
                logger.info("Inference request completed request_id=%s", request_id)

            self._request_store.update_queue_positions(
                active_request_id=self._active_request_id,
                queued_request_ids=list(self._queued_request_ids),
            )
            self._condition.notify_all()

    def update_progress(
        self,
        request_id: str,
        *,
        status: str | None = None,
        message: str | None = None,
        response_preview: str | None = None,
    ) -> None:
        self._request_store.update_progress(
            request_id,
            status=status,
            message=message,
            response_preview=response_preview,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            active_request = (
                self._request_store.get_request(self._active_request_id)
                if self._active_request_id
                else None
            )
            queued_requests = [
                self._request_store.get_request(request_id)
                for request_id in self._queued_request_ids
            ]
        queued_requests = [request for request in queued_requests if request is not None]
        return {
            "active_request_id": self._active_request_id,
            "active_request": active_request,
            "queued_count": len(queued_requests),
            "queued_requests": queued_requests,
        }


class LlamaCppServerRuntime:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._current_model_key: str | None = None
        self._current_quantization_key: str | None = None
        try:
            self._cleanup_stale_processes()
        except Exception:
            logger.exception("Initial llama.cpp stale process cleanup failed")

    def _cleanup_stale_processes(self, *, exclude_pid: int | None = None) -> None:
        candidate_pids = get_listening_pids_for_port(LLAMA_SERVER_PORT) | get_llama_server_process_ids()
        if exclude_pid is not None:
            candidate_pids.discard(exclude_pid)
        if not candidate_pids:
            return
        logger.warning(
            "Cleaning stale llama.cpp processes pids=%s port=%s",
            sorted(candidate_pids),
            LLAMA_SERVER_PORT,
        )
        kill_process_ids(candidate_pids, exclude_pid=exclude_pid)

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
            self._cleanup_stale_processes()
            log_runtime_memory_snapshot("llama.cpp-unload-no-tracked-process")
            return

        if process.poll() is None:
            logger.info("Terminating tracked llama.cpp process pid=%s", process.pid)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing tracked llama.cpp process pid=%s after timeout", process.pid)
                process.kill()
                process.wait(timeout=5)

        self._cleanup_stale_processes()
        log_runtime_memory_snapshot("llama.cpp-unload-finished")

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
            "--cache-ram",
            str(max(0, LLAMA_SERVER_CACHE_RAM_MIB)),
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
        log_runtime_memory_snapshot(
            f"llama.cpp-load-before-unload:{spec['key']}:{quantization_spec['key']}"
        )
        self.unload()
        self._cleanup_stale_processes()
        command = self._build_command(spec, quantization_spec)
        logger.info(
            "Launching llama.cpp model key=%s quantization=%s command=%s",
            spec["key"],
            quantization_spec["key"],
            command,
        )
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

        emit(76, "Waiting for llama.cpp to finish loading the quantized model...")
        try:
            self._wait_until_ready()
        except Exception:
            logger.exception(
                "llama.cpp process failed during readiness key=%s quantization=%s pid=%s",
                spec["key"],
                quantization_spec["key"],
                process.pid,
            )
            self.unload()
            raise

        with self._process_lock:
            if self._process is process and process.poll() is None:
                self._current_model_key = spec["key"]
                self._current_quantization_key = quantization_spec["key"]

        logger.info(
            "llama.cpp model ready key=%s quantization=%s pid=%s",
            spec["key"],
            quantization_spec["key"],
            process.pid,
        )
        log_runtime_memory_snapshot(
            f"llama.cpp-load-ready:{spec['key']}:{quantization_spec['key']}"
        )
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
        except ConnectionResetError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The llama.cpp runtime reset the connection while handling this turn. "
                    "This usually means the media payload was rejected by the underlying "
                    "runtime or the process crashed during multimodal preprocessing. "
                    "Check logs/llama-server.log for the exact cause."
                ),
            ) from exc
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
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_image_to_data_url(image)},
                    },
                ],
            }
        )
        return messages

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
        payload = {
            "messages": self._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
                image=image,
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
        finish_reason = choice.get("finish_reason") or "stop"
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
            "finish_reason": finish_reason,
            "hit_max_tokens": finish_reason == "length",
            "max_new_tokens_requested": max_new_tokens,
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
        payload = {
            "messages": self._build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
                image=image,
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
        finish_reason = "stop"

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
                    finish_reason = choice.get("finish_reason") or finish_reason
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
        except ConnectionResetError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The llama.cpp runtime reset the connection while streaming this turn. "
                    "Check logs/llama-server.log for the exact multimodal failure."
                ),
            ) from exc
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
            "finish_reason": finish_reason,
            "hit_max_tokens": finish_reason == "length",
            "max_new_tokens_requested": max_new_tokens,
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
        finish_reason = choice.get("finish_reason") or "stop"
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
            "finish_reason": finish_reason,
            "hit_max_tokens": finish_reason == "length",
            "max_new_tokens_requested": max_new_tokens,
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
        finish_reason = "stop"

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
                    finish_reason = choice.get("finish_reason") or finish_reason
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
            "finish_reason": finish_reason,
            "hit_max_tokens": finish_reason == "length",
            "max_new_tokens_requested": max_new_tokens,
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
            target_model_key=None,
            target_quantization_key=None,
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

    def current_selection(self) -> dict | None:
        self._reconcile_runtime_state()
        if (
            not self.is_loaded
            or self._current_model_key is None
            or self._current_quantization_key is None
        ):
            return None

        spec = get_model_spec(self._current_model_key)
        quantization_spec = get_quantization_spec(self._current_quantization_key)
        return {
            "model_key": spec["key"],
            "quantization_key": quantization_spec["key"],
            "model": serialize_model_spec(spec),
            "quantization": serialize_quantization_spec(quantization_spec),
            "runtime_capabilities": build_runtime_capabilities(spec, quantization_spec),
            "runtime_family": self._runtime_family,
            "loaded_at": self.loaded_at,
        }

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
        logger.info(
            "Unloading current model runtime_family=%s model_key=%s quantization=%s",
            self._runtime_family,
            self._current_model_key,
            self._current_quantization_key,
        )
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
        log_runtime_memory_snapshot("service-unload-finished")

    def unload(self) -> dict:
        load_state = self.get_load_state()
        if load_state["is_loading"]:
            target_model = load_state.get("target_model") or {}
            target_quantization = load_state.get("target_quantization") or {}
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{target_model.get('label', 'A model')} / "
                    f"{target_quantization.get('label', 'the selected quantization')} is "
                    "still loading. Wait for it to finish before unloading."
                ),
            )

        previous = self.current_selection()
        with self._load_lock:
            self._unload_current_model()
            self._set_load_state(
                status="idle",
                progress=0,
                message="No model is loaded.",
                target_model_key=None,
                target_quantization_key=None,
                error=None,
                started_at=None,
                finished_at=None,
            )

        return {
            "unloaded": previous is not None,
            "previous": previous,
            "health": self.health(),
        }

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
        requested_max_new_tokens: int,
    ) -> dict:
        generated_tokens = int(outputs[0].shape[-1] - input_len)
        raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed = processor.parse_response(raw_response)
        parsed_reply = extract_reply(parsed)
        finish_reason, hit_max_tokens = infer_transformers_finish_reason(
            processor, outputs, input_len, requested_max_new_tokens
        )
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
            "finish_reason": finish_reason,
            "hit_max_tokens": hit_max_tokens,
            "max_new_tokens_requested": requested_max_new_tokens,
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
            "runtime_capabilities": build_runtime_capabilities(
                active_spec, active_quantization
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
        tools: list[dict[str, Any]] | None = None,
        override_messages: list[dict[str, Any]] | None = None,
    ) -> dict:
        processor, model, spec, quantization_spec = self.require_loaded_selection(
            model_key, quantization_key
        )
        if self._runtime_family == "llama.cpp":
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
                image=image,
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
        messages = override_messages or self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            image=image,
            audio=audio,
        )

        start_time = time.perf_counter()
        with self._generate_lock:
            chat_template_kwargs = {
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "add_generation_prompt": True,
                "enable_thinking": thinking,
            }
            if tools:
                chat_template_kwargs["tools"] = tools
            inputs = processor.apply_chat_template(messages, **chat_template_kwargs).to(
                model.device
            )
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
            requested_max_new_tokens=max_new_tokens,
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
        tools: list[dict[str, Any]] | None = None,
        override_messages: list[dict[str, Any]] | None = None,
    ):
        processor, model, spec, quantization_spec = self.require_loaded_selection(
            model_key, quantization_key
        )
        if self._runtime_family == "llama.cpp":
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
                image=image,
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
        messages = override_messages or self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            image=image,
            audio=audio,
        )

        start_time = time.perf_counter()
        with self._generate_lock:
            chat_template_kwargs = {
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "add_generation_prompt": True,
                "enable_thinking": thinking,
            }
            if tools:
                chat_template_kwargs["tools"] = tools
            inputs = processor.apply_chat_template(messages, **chat_template_kwargs).to(
                model.device
            )
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
            requested_max_new_tokens=max_new_tokens,
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
        if answer is None and content is None and parsed.get("tool_calls"):
            return {
                "reply": "",
                "thought": thought,
            }
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


def load_image_bytes(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unsupported image file.") from exc

    min_width = max(2, image.width)
    min_height = max(2, image.height)
    if min_width != image.width or min_height != image.height:
        logger.info(
            "Upscaling tiny image for multimodal runtime original=%sx%s resized=%sx%s",
            image.width,
            image.height,
            min_width,
            min_height,
        )
        image = image.resize((min_width, min_height), Image.Resampling.BICUBIC)

    return image


def load_image(upload: UploadFile | None) -> Image.Image | None:
    if upload is None:
        return None
    return load_image_bytes(upload.file.read())


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


def load_audio_bytes(data: bytes, target_rate: int) -> np.ndarray:
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail="Unsupported audio file.") from exc

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    return resample_audio(audio, sample_rate, target_rate)


def load_audio(upload: UploadFile | None, target_rate: int) -> np.ndarray | None:
    if upload is None:
        return None
    return load_audio_bytes(upload.file.read(), target_rate)


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


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def parse_json_fragment(text: str) -> Any | None:
    cleaned = strip_code_fence(text)
    if not cleaned:
        return None

    decoder = json.JSONDecoder()
    for start_index, character in enumerate(cleaned):
        if character not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[start_index:])
            return parsed
        except json.JSONDecodeError:
            continue
    return None


def read_binary_from_source(source: str, *, media_label: str) -> bytes:
    normalized = (source or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail=f"Empty {media_label} source.")

    if normalized.startswith("data:"):
        try:
            header, encoded = normalized.split(",", 1)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Malformed {media_label} data URL.",
            ) from exc
        if ";base64" not in header.lower():
            raise HTTPException(
                status_code=400,
                detail=f"{media_label.capitalize()} data URLs must be base64 encoded.",
            )
        try:
            return base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 payload for {media_label}.",
            ) from exc

    parsed_url = urllib.parse.urlparse(normalized)
    if parsed_url.scheme in {"http", "https"}:
        request = urllib.request.Request(
            normalized,
            method="GET",
            headers={"User-Agent": "gemma4-lab-local-api/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not fetch {media_label} from {normalized}.",
            ) from exc

    if parsed_url.scheme == "file":
        raw_path = urllib.request.url2pathname(urllib.parse.unquote(parsed_url.path))
        if re.match(r"^/[A-Za-z]:/", raw_path):
            raw_path = raw_path[1:]
        if parsed_url.netloc:
            raw_path = f"//{parsed_url.netloc}{raw_path}"
        path = Path(raw_path)
    else:
        path = Path(normalized)
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()

    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"{media_label.capitalize()} source does not exist: {path}",
        )

    try:
        return path.read_bytes()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {media_label} source: {path}",
        ) from exc


def normalize_api_tool_choice(
    tool_choice: str | ApiToolChoiceObject | None,
) -> dict[str, str | None]:
    if isinstance(tool_choice, ApiToolChoiceObject):
        return {"mode": "function", "name": tool_choice.function.name}

    normalized = (tool_choice or "auto").strip().lower() if isinstance(tool_choice, str) else "auto"
    if normalized in {"auto", "none", "required"}:
        return {"mode": normalized, "name": None}
    return {"mode": "auto", "name": None}


def normalize_api_tools(tools: list[ApiToolDefinition] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools or []:
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description or "",
                    "parameters": tool.function.parameters
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return normalized


def build_json_response_instruction(response_format: ApiResponseFormat | None) -> str:
    if response_format is None or response_format.type != "json_object":
        return ""

    lines = [
        "Return only a valid JSON object.",
        "Do not use markdown fences, prose before the JSON, or prose after the JSON.",
    ]
    if response_format.json_schema:
        lines.append("Follow this JSON Schema as closely as possible:")
        lines.append(json.dumps(response_format.json_schema, ensure_ascii=False))
    return "\n".join(lines)


def build_tool_prompt_instruction(
    tools: list[dict[str, Any]],
    tool_choice: dict[str, str | None],
) -> str:
    if not tools:
        return ""

    lines = [
        "You have access to functions.",
        (
            "If you decide to call a function, return JSON only. Use either "
            '{"name":"function_name","parameters":{...}} for one call or '
            '{"tool_calls":[{"name":"function_name","parameters":{...}}]} for '
            "multiple calls."
        ),
        "Do not add commentary around a function call response.",
    ]
    if tool_choice["mode"] == "none":
        lines.insert(0, "Do not call any function for this turn. Answer directly.")
    elif tool_choice["mode"] == "required":
        lines.insert(0, "You must call at least one function for this turn.")
    elif tool_choice["mode"] == "function" and tool_choice["name"]:
        lines.insert(
            0,
            f'If you call a function for this turn, it must be "{tool_choice["name"]}".',
        )

    tool_definitions = [tool["function"] for tool in tools]
    lines.append(json.dumps(tool_definitions, ensure_ascii=False))
    return "\n".join(lines)


def merge_system_prompt_sections(*sections: str) -> str:
    normalized = [section.strip() for section in sections if section and section.strip()]
    return "\n\n".join(normalized).strip() or DEFAULT_SYSTEM_PROMPT


def parse_api_message_content(
    content: str | list[ApiMessageContentPart],
    *,
    target_audio_rate: int,
) -> tuple[str, Image.Image | None, np.ndarray | None]:
    if isinstance(content, str):
        return content.strip(), None, None

    text_parts: list[str] = []
    image_payload: Image.Image | None = None
    audio_payload: np.ndarray | None = None

    for part in content:
        if part.type == "text":
            if part.text and part.text.strip():
                text_parts.append(part.text.strip())
            continue

        if part.type == "image_url":
            if image_payload is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Only one image is supported per chat completion request.",
                )
            if part.image_url is None or not part.image_url.url.strip():
                raise HTTPException(status_code=400, detail="Image URL is missing.")
            image_payload = load_image_bytes(
                read_binary_from_source(part.image_url.url, media_label="image")
            )
            continue

        if part.type == "audio_url":
            if audio_payload is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Only one audio clip is supported per chat completion request.",
                )
            if part.audio_url is None or not part.audio_url.url.strip():
                raise HTTPException(status_code=400, detail="Audio URL is missing.")
            audio_payload = load_audio_bytes(
                read_binary_from_source(part.audio_url.url, media_label="audio"),
                target_audio_rate,
            )

    return "\n\n".join(text_parts).strip(), image_payload, audio_payload


def normalize_function_arguments(arguments: Any) -> tuple[str, Any]:
    if isinstance(arguments, str):
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_arguments = None
        return arguments, parsed_arguments

    normalized_arguments = arguments if isinstance(arguments, dict) else {}
    return json.dumps(normalized_arguments, ensure_ascii=False), normalized_arguments


def normalize_tool_call_items(candidate: Any) -> list[dict[str, Any]]:
    if candidate is None:
        return []

    if isinstance(candidate, dict):
        if isinstance(candidate.get("tool_calls"), list):
            return normalize_tool_call_items(candidate["tool_calls"])

        if candidate.get("type") == "function" and isinstance(candidate.get("function"), dict):
            function_payload = candidate["function"]
            function_name = function_payload.get("name")
            if function_name:
                arguments_text, parsed_arguments = normalize_function_arguments(
                    function_payload.get("arguments", {})
                )
                return [
                    {
                        "id": candidate.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments_text,
                        },
                        "parsed_arguments": parsed_arguments,
                    }
                ]

        function_name = candidate.get("name")
        if function_name:
            arguments_text, parsed_arguments = normalize_function_arguments(
                candidate.get("parameters", candidate.get("arguments", candidate.get("args", {})))
            )
            return [
                {
                    "id": candidate.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": arguments_text,
                    },
                    "parsed_arguments": parsed_arguments,
                }
            ]
        return []

    if isinstance(candidate, list):
        normalized: list[dict[str, Any]] = []
        for item in candidate:
            normalized.extend(normalize_tool_call_items(item))
        return normalized

    return []


def infer_tool_calls_from_response(parsed: Any, raw_response: str, reply: str) -> list[dict[str, Any]]:
    tool_calls = normalize_tool_call_items(parsed)
    if tool_calls:
        return tool_calls

    parsed_json = parse_json_fragment(raw_response) or parse_json_fragment(reply)
    return normalize_tool_call_items(parsed_json)


def build_local_api_capabilities() -> dict[str, Any]:
    return {
        "base_url": "http://127.0.0.1:8000",
        "control_routes": {
            "health": "/api/v1/health",
            "status": "/api/v1/status",
            "models": "/api/v1/models",
            "current_model": "/api/v1/models/current",
            "load_model": "/api/v1/models/load",
            "unload_model": "/api/v1/models/unload",
            "load_status": "/api/v1/models/load-status",
            "capabilities": "/api/v1/capabilities",
            "monitoring": "/api/v1/monitoring",
            "requests": "/api/v1/requests",
            "async_chat_submit": "/api/v1/requests/chat/completions",
        },
        "openai_compatible_routes": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
        },
        "structured_output": {
            "text": {"supported": True},
            "json_object": {
                "supported": True,
                "mode": "best_effort",
                "notes": (
                    "Gemma supports prompt-based structured output. This gateway can ask for "
                    "JSON-only output and validates it after generation, but it does not "
                    "provide hard constrained decoding."
                ),
            },
        },
        "tool_calling": {
            "supported": True,
            "execution_supported": False,
            "native_transformers_support": True,
            "other_runtimes_mode": "prompted_best_effort",
            "notes": (
                "Gemma can emit function-call-shaped text, but the caller must execute the "
                "tool. The Transformers path can also use the official chat template "
                "tool definitions."
            ),
        },
        "multimodal_input": {
            "latest_user_turn_only": True,
            "accepted_content_parts": ["text", "image_url", "audio_url"],
            "supported_media_sources": ["data_url", "file_url", "absolute_path", "http_url"],
            "audio_runtime_note": "Audio input only works on E2B/E4B and only on the Transformers path.",
        },
        "request_queue": {
            "enabled": True,
            "persistence": "sqlite",
            "batching": {
                "enabled": False,
                "notes": (
                    "This lab currently serializes GPU inference requests across runtimes. "
                    "Mixed local runtimes and multimodal payloads are tracked in a unified "
                    "queue, but not batch-merged."
                ),
            },
        },
    }


def resolve_api_model_selection(
    request: ApiChatCompletionRequest,
) -> tuple[dict, dict]:
    selected_model_key = request.model_key
    selected_quantization_key = request.quantization_key

    if request.model:
        model_reference = request.model.strip()
        if ":" in model_reference:
            model_reference, inferred_quantization = model_reference.rsplit(":", 1)
            if not selected_quantization_key:
                selected_quantization_key = inferred_quantization.strip()

        lowered_reference = model_reference.strip().lower()
        matched_spec = None
        for spec in MODEL_SPECS:
            if lowered_reference in {
                spec["key"],
                spec["hf_model_id"].lower(),
                spec["label"].lower(),
            }:
                matched_spec = spec
                break

        if matched_spec is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model reference: {request.model}",
            )
        selected_model_key = matched_spec["key"]

    current_selection = service.current_selection()
    if current_selection is not None:
        if not selected_model_key:
            selected_model_key = current_selection["model_key"]
        if not selected_quantization_key:
            selected_quantization_key = current_selection["quantization_key"]

    spec = get_model_spec(selected_model_key or DEFAULT_MODEL_KEY)
    quantization_spec = get_quantization_spec(
        selected_quantization_key or DEFAULT_QUANTIZATION_KEY
    )
    return spec, quantization_spec


def prepare_api_chat_completion_request(
    request: ApiChatCompletionRequest,
) -> dict[str, Any]:
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")

    spec, quantization_spec = resolve_api_model_selection(request)
    processor, _, _, _ = service.require_loaded_selection(
        spec["key"], quantization_spec["key"]
    )
    target_audio_rate = getattr(
        getattr(processor, "feature_extractor", None),
        "sampling_rate",
        16000,
    )
    normalized_tools = normalize_api_tools(request.tools)
    tool_choice = normalize_api_tool_choice(request.tool_choice)

    final_message = request.messages[-1]
    if final_message.role != "user":
        raise HTTPException(
            status_code=400,
            detail="The last message must be a user message for this local chat endpoint.",
        )

    system_sections: list[str] = []
    history: list[HistoryTurn] = []
    runtime_messages: list[dict[str, Any]] = []

    prompt = ""
    image_payload: Image.Image | None = None
    audio_payload: np.ndarray | None = None

    for index, message in enumerate(request.messages):
        text_content, message_image, message_audio = parse_api_message_content(
            message.content,
            target_audio_rate=target_audio_rate,
        )

        if message.role == "system":
            if text_content:
                system_sections.append(text_content)
                runtime_messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": text_content}],
                    }
                )
            continue

        is_last_turn = index == len(request.messages) - 1
        if is_last_turn:
            prompt = text_content
            image_payload = message_image
            audio_payload = message_audio
            latest_content: list[dict[str, Any]] = []
            if message_image is not None:
                latest_content.append({"type": "image", "image": message_image})
            if message_audio is not None:
                latest_content.append({"type": "audio", "audio": message_audio})
            latest_content.append({"type": "text", "text": prompt})
            runtime_messages.append({"role": "user", "content": latest_content})
            continue

        history_text = text_content.strip()
        if message.role in {"user", "assistant"} and history_text:
            history.append(HistoryTurn(role=message.role, content=history_text))
            runtime_messages.append(
                {
                    "role": message.role,
                    "content": [{"type": "text", "text": history_text}],
                }
            )
        elif message.role == "tool" and history_text:
            tool_context = f"Tool result:\n{history_text}"
            history.append(HistoryTurn(role="assistant", content=tool_context))
            runtime_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": tool_context}],
                }
            )

    if image_payload is not None and not spec["supports_image"]:
        raise HTTPException(
            status_code=400,
            detail=f"{spec['label']} does not support image input in the official tables.",
        )
    if audio_payload is not None and not spec["supports_audio"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{spec['label']} does not support audio input according to the official "
                "Gemma 4 Supported Modalities tables."
            ),
        )
    if quantization_spec.get("runtime_family") in {"llama.cpp", "vllm-wsl"} and audio_payload is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{spec['label']} in {quantization_spec['label']} does not support audio "
                "input in this runtime path."
            ),
        )

    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        if image_payload is not None and audio_payload is not None:
            normalized_prompt = "Describe the image and transcribe the audio."
        elif image_payload is not None:
            normalized_prompt = "Describe this image."
        elif audio_payload is not None:
            normalized_prompt = "Transcribe this audio."
        else:
            raise HTTPException(
                status_code=400,
                detail="A user text prompt, image, or audio clip is required.",
            )

        last_runtime_message = runtime_messages[-1]
        last_runtime_message["content"][-1]["text"] = normalized_prompt

    runtime_family = quantization_spec.get("runtime_family")
    system_prompt = merge_system_prompt_sections(
        *system_sections,
        build_json_response_instruction(request.response_format),
        (
            build_tool_prompt_instruction(normalized_tools, tool_choice)
            if runtime_family != "transformers"
            else ""
        ),
    )

    if runtime_messages and runtime_messages[0]["role"] == "system":
        runtime_messages[0]["content"] = [{"type": "text", "text": system_prompt}]
    else:
        runtime_messages.insert(
            0,
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        )

    return {
        "spec": spec,
        "quantization_spec": quantization_spec,
        "prompt": normalized_prompt,
        "system_prompt": system_prompt,
        "history": history[-8:],
        "image": image_payload,
        "audio": audio_payload,
        "tools": normalized_tools,
        "tool_choice": tool_choice,
        "runtime_messages": runtime_messages,
    }


def build_chat_completion_usage(response_payload: dict[str, Any]) -> dict[str, int] | None:
    prompt_tokens = response_payload.get("prompt_tokens")
    completion_tokens = response_payload.get("generated_tokens")
    if not isinstance(prompt_tokens, int) and not isinstance(completion_tokens, int):
        return None
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(prompt_tokens or 0) + int(completion_tokens or 0),
    }


def build_chat_completion_response(
    *,
    request: ApiChatCompletionRequest,
    prepared: dict[str, Any],
    response_payload: dict[str, Any],
    request_id: str | None = None,
    completion_id: str | None = None,
    created_at: int | None = None,
) -> dict[str, Any]:
    completion_id = completion_id or f"chatcmpl_{uuid.uuid4().hex}"
    created_at = created_at or int(time.time())
    tool_calls = infer_tool_calls_from_response(
        response_payload.get("parsed"),
        str(response_payload.get("raw_response") or ""),
        str(response_payload.get("reply") or ""),
    )
    finish_reason = str(response_payload.get("finish_reason") or "stop")
    if tool_calls and not str(response_payload.get("reply") or "").strip():
        finish_reason = "tool_calls"

    structured_output = None
    structured_output_error = None
    if request.response_format is not None and request.response_format.type == "json_object":
        parsed_json = parse_json_fragment(str(response_payload.get("raw_response") or ""))
        if parsed_json is None:
            parsed_json = parse_json_fragment(str(response_payload.get("reply") or ""))
        if isinstance(parsed_json, dict):
            structured_output = parsed_json
        else:
            structured_output_error = "The model did not return a valid JSON object."

    usage = build_chat_completion_usage(response_payload)
    model_id = (
        f"{prepared['spec']['key']}:{prepared['quantization_spec']['key']}"
    )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response_payload.get("reply") or "",
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "request_id": request_id,
        "status_url": f"/api/v1/requests/{request_id}" if request_id else None,
        "id": completion_id,
        "object": "chat.completion",
        "created": created_at,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
        "structured_output": structured_output,
        "structured_output_error": structured_output_error,
        "gemma_lab": {
            "request_id": request_id,
            "status_url": f"/api/v1/requests/{request_id}" if request_id else None,
            "runtime_family": service.health()["runtime_family"],
            "active_model_key": response_payload.get("active_model_key"),
            "active_quantization_key": response_payload.get("active_quantization_key"),
            "thought": response_payload.get("thought"),
            "raw_response": response_payload.get("raw_response"),
            "elapsed_ms": response_payload.get("elapsed_ms"),
            "hit_max_tokens": response_payload.get("hit_max_tokens"),
            "max_new_tokens_requested": response_payload.get("max_new_tokens_requested"),
            "tts_audio": response_payload.get("tts_audio"),
            "tts_audio_error": response_payload.get("tts_audio_error"),
        },
    }


def encode_sse_payload(payload: dict[str, Any] | str) -> bytes:
    if isinstance(payload, str):
        body = payload
    else:
        body = json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n".encode("utf-8")


def summarize_prompt_text(text: str, *, limit: int = 280) -> str:
    normalized = WHITESPACE_PATTERN.sub(" ", (text or "").strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def build_request_payload_summary(
    *,
    route: str,
    model_key: str,
    quantization_key: str,
    runtime_family: str | None,
    prompt: str,
    system_prompt: str,
    history_turns: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    thinking: bool,
    has_image: bool,
    has_audio: bool,
    stream: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "route": route,
        "model_key": model_key,
        "quantization_key": quantization_key,
        "runtime_family": runtime_family,
        "prompt_preview": summarize_prompt_text(prompt),
        "system_prompt_preview": summarize_prompt_text(system_prompt, limit=180),
        "history_turns": history_turns,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "thinking": thinking,
        "has_image": has_image,
        "has_audio": has_audio,
        "stream": stream,
        "prompt_chars": len(prompt or ""),
        "system_prompt_chars": len(system_prompt or ""),
    }
    if extra:
        payload.update(extra)
    return payload


def build_monitoring_snapshot(*, recent_limit: int = 24) -> dict[str, Any]:
    commit_snapshot = get_windows_commit_snapshot()
    gpu_snapshot = get_gpu_monitor_snapshot()
    queue_snapshot = request_queue.snapshot()
    return {
        "health": service.health(),
        "current_model": service.current_selection(),
        "gpu": gpu_snapshot,
        "memory": commit_snapshot,
        "queue": queue_snapshot,
        "recent_requests": request_store.list_requests(limit=recent_limit),
        "database": {
            "path": str(REQUESTS_DB_PATH),
        },
    }


request_store = InferenceRequestStore(REQUESTS_DB_PATH)
request_queue = InferenceQueueManager(request_store)
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


@app.get("/api/v1/health")
def api_v1_health() -> dict:
    return service.health()


@app.get("/api/v1/status")
def api_v1_status() -> dict:
    return {
        "health": service.health(),
        "current_model": service.current_selection(),
        "queue": request_queue.snapshot(),
        "capabilities": build_local_api_capabilities(),
    }


@app.get("/api/v1/capabilities")
def api_v1_capabilities() -> dict:
    return build_local_api_capabilities()


@app.get("/api/v1/models")
def api_v1_models() -> dict:
    return {
        "current_model": service.current_selection(),
        "load_state": service.get_load_state(),
        "models": [serialize_model_spec(spec) for spec in MODEL_SPECS],
        "quantizations": [serialize_quantization_spec(spec) for spec in QUANTIZATION_SPECS],
    }


@app.get("/api/v1/models/current")
def api_v1_current_model() -> dict:
    return {
        "loaded": service.is_loaded,
        "current_model": service.current_selection(),
        "load_state": service.get_load_state(),
    }


@app.get("/api/v1/models/load-status")
def api_v1_model_load_status() -> dict:
    return {
        "load_state": service.get_load_state(),
        "current_model": service.current_selection(),
    }


@app.get("/api/v1/monitoring")
def api_v1_monitoring() -> dict:
    return build_monitoring_snapshot()


@app.get("/api/v1/requests")
def api_v1_list_requests(limit: int = 40) -> dict:
    return {
        "requests": request_store.list_requests(limit=max(1, min(limit, 200))),
        "queue": request_queue.snapshot(),
    }


@app.get("/api/v1/requests/{request_id}")
def api_v1_get_request(request_id: str) -> dict:
    request_row = request_store.get_request(request_id)
    if request_row is None:
        raise HTTPException(status_code=404, detail="Request ID not found.")
    return {
        "request": request_row,
        "queue": request_queue.snapshot(),
    }


@app.post("/api/v1/models/load")
def api_v1_load_model(request: ModelLoadRequest) -> dict:
    logger.info(
        "External API model load requested key=%s quantization=%s",
        request.model_key,
        request.quantization_key,
    )
    load_result = service.start_load(request.model_key, request.quantization_key)
    return {
        "accepted": load_result["accepted"],
        "message": load_result["message"],
        "load_state": load_result["load_state"],
        "current_model": service.current_selection(),
        "health": service.health(),
    }


@app.post("/api/v1/models/unload")
def api_v1_unload_model() -> dict:
    logger.info("External API unload requested")
    return service.unload()


@app.get("/v1/models")
def openai_compatible_models() -> dict:
    current_selection = service.current_selection()
    data = []
    for model_spec in MODEL_SPECS:
        supported_quantization_keys = [
            key
            for key in model_spec["memory_requirements_gib"].keys()
            if key in QUANTIZATION_SPECS_BY_KEY
        ]
        if model_spec["key"] == "31b-nvfp4" and "nvfp4" not in supported_quantization_keys:
            supported_quantization_keys.append("nvfp4")

        for quantization_key in supported_quantization_keys:
            quantization_spec = get_quantization_spec(quantization_key)
            data.append(
                {
                    "id": f"{model_spec['key']}:{quantization_spec['key']}",
                    "object": "model",
                    "created": 0,
                    "owned_by": "gemma4-lab",
                    "root": model_spec["hf_model_id"],
                    "permission": [],
                    "runtime_family": quantization_spec["runtime_family"],
                    "loaded": (
                        current_selection is not None
                        and current_selection["model_key"] == model_spec["key"]
                        and current_selection["quantization_key"] == quantization_spec["key"]
                    ),
                }
            )

    return {"object": "list", "data": data}


@app.post("/api/v1/chat/completions")
@app.post("/v1/chat/completions")
def api_v1_chat_completions(request: ApiChatCompletionRequest):
    prepared = prepare_api_chat_completion_request(request)
    spec = prepared["spec"]
    quantization_spec = prepared["quantization_spec"]
    request_id = request_queue.register_request(
        route="/v1/chat/completions",
        runtime_family=quantization_spec.get("runtime_family"),
        model_key=spec["key"],
        quantization_key=quantization_spec["key"],
        stream=request.stream,
        request_payload=build_request_payload_summary(
            route="/v1/chat/completions",
            model_key=spec["key"],
            quantization_key=quantization_spec["key"],
            runtime_family=quantization_spec.get("runtime_family"),
            prompt=prepared["prompt"],
            system_prompt=prepared["system_prompt"],
            history_turns=len(prepared["history"]),
            max_new_tokens=max(32, min(int(request.max_tokens), 1024)),
            temperature=max(0.0, min(float(request.temperature), 2.0)),
            top_p=max(0.1, min(float(request.top_p), 1.0)),
            top_k=max(1, min(int(request.top_k), 128)),
            thinking=request.thinking,
            has_image=prepared["image"] is not None,
            has_audio=prepared["audio"] is not None,
            stream=request.stream,
            extra={
                "tool_count": len(prepared["tools"]),
                "response_format": request.response_format.type
                if request.response_format
                else "text",
            },
        ),
        request_id=request.request_id,
    )
    logger.info(
        "External chat completion requested request_id=%s key=%s quantization=%s stream=%s thinking=%s json_mode=%s tools=%s prompt_chars=%s history_turns=%s image=%s audio=%s",
        request_id,
        spec["key"],
        quantization_spec["key"],
        request.stream,
        request.thinking,
        request.response_format.type if request.response_format else "text",
        len(prepared["tools"]),
        len(prepared["prompt"]),
        len(prepared["history"]),
        prepared["image"] is not None,
        prepared["audio"] is not None,
    )

    runtime_messages = (
        prepared["runtime_messages"]
        if quantization_spec.get("runtime_family") == "transformers"
        else None
    )

    if request.stream:
        completion_id = f"chatcmpl_{uuid.uuid4().hex}"
        created_at = int(time.time())
        model_id = f"{spec['key']}:{quantization_spec['key']}"

        def event_stream():
            preview_text = ""
            try:
                request_queue.wait_for_turn(
                    request_id,
                    message="Generating streamed response.",
                )
                yield encode_sse_payload(
                    {
                        "request_id": request_id,
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_at,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                for event in service.stream_generate(
                    model_key=spec["key"],
                    quantization_key=quantization_spec["key"],
                    prompt=prepared["prompt"],
                    system_prompt=prepared["system_prompt"],
                    history=prepared["history"],
                    image=prepared["image"],
                    audio=prepared["audio"],
                    max_new_tokens=max(32, min(int(request.max_tokens), 1024)),
                    temperature=max(0.0, min(float(request.temperature), 2.0)),
                    top_p=max(0.1, min(float(request.top_p), 1.0)),
                    top_k=max(1, min(int(request.top_k), 128)),
                    thinking=request.thinking,
                    tts_enabled=request.tts_enabled,
                    tools=prepared["tools"],
                    override_messages=runtime_messages,
                ):
                    if event.get("event") == "token":
                        preview_text = summarize_prompt_text(
                            preview_text + event["text"],
                            limit=500,
                        )
                        request_queue.update_progress(
                            request_id,
                            status="running",
                            message="Streaming tokens.",
                            response_preview=preview_text,
                        )
                        yield encode_sse_payload(
                            {
                                "request_id": request_id,
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_at,
                                "model": model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": event["text"]},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                        continue

                    if event.get("event") != "done":
                        continue

                    final_payload = build_chat_completion_response(
                        request=request,
                        prepared=prepared,
                        response_payload=event["payload"],
                        request_id=request_id,
                        completion_id=completion_id,
                        created_at=created_at,
                    )
                    request_queue.finish(
                        request_id,
                        response_payload=final_payload,
                        response_preview=summarize_prompt_text(
                            final_payload["choices"][0]["message"].get("content", ""),
                            limit=500,
                        ),
                    )
                    final_choice = {
                        "index": 0,
                        "delta": {},
                        "finish_reason": final_payload["choices"][0]["finish_reason"],
                    }
                    tool_calls = final_payload["choices"][0]["message"].get("tool_calls")
                    if tool_calls:
                        final_choice["delta"]["tool_calls"] = tool_calls
                    yield encode_sse_payload(
                        {
                            "request_id": request_id,
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_at,
                            "model": model_id,
                            "choices": [final_choice],
                            "usage": final_payload.get("usage"),
                            "structured_output": final_payload.get("structured_output"),
                            "structured_output_error": final_payload.get(
                                "structured_output_error"
                            ),
                            "gemma_lab": final_payload.get("gemma_lab"),
                        }
                    )
                    break
            except HTTPException as exc:
                request_queue.finish(request_id, error_text=str(exc.detail))
                yield encode_sse_payload(
                    {
                        "request_id": request_id,
                        "error": {
                            "message": str(exc.detail),
                            "type": "HTTPException",
                            "code": exc.status_code,
                        }
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception(
                    "External chat completion stream failed key=%s quantization=%s",
                    spec["key"],
                    quantization_spec["key"],
                )
                request_queue.finish(
                    request_id,
                    error_text=render_model_load_error(spec, exc),
                )
                yield encode_sse_payload(
                    {
                        "request_id": request_id,
                        "error": {
                            "message": render_model_load_error(spec, exc),
                            "type": exc.__class__.__name__,
                            "code": 500,
                        }
                    }
                )
            yield encode_sse_payload("[DONE]")

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        request_queue.wait_for_turn(
            request_id,
            message="Generating response.",
        )
        response_payload = service.generate(
            model_key=spec["key"],
            quantization_key=quantization_spec["key"],
            prompt=prepared["prompt"],
            system_prompt=prepared["system_prompt"],
            history=prepared["history"],
            image=prepared["image"],
            audio=prepared["audio"],
            max_new_tokens=max(32, min(int(request.max_tokens), 1024)),
            temperature=max(0.0, min(float(request.temperature), 2.0)),
            top_p=max(0.1, min(float(request.top_p), 1.0)),
            top_k=max(1, min(int(request.top_k), 128)),
            thinking=request.thinking,
            tools=prepared["tools"],
            override_messages=runtime_messages,
        )
        if request.tts_enabled and response_payload.get("reply"):
            try:
                response_payload["tts_audio"] = tts_service.synthesize(response_payload["reply"])
            except Exception:
                logger.exception(
                    "TTS synthesis failed for external chat completion key=%s quantization=%s",
                    spec["key"],
                    quantization_spec["key"],
                )
                response_payload["tts_audio"] = None
                response_payload["tts_audio_error"] = "Local TTS synthesis failed."

        final_payload = build_chat_completion_response(
            request=request,
            prepared=prepared,
            response_payload=response_payload,
            request_id=request_id,
        )
        request_queue.finish(
            request_id,
            response_payload=final_payload,
            response_preview=summarize_prompt_text(
                final_payload["choices"][0]["message"].get("content", ""),
                limit=500,
            ),
        )
        return final_payload
    except HTTPException as exc:
        request_queue.finish(request_id, error_text=str(exc.detail))
        raise
    except Exception as exc:
        request_queue.finish(request_id, error_text=render_model_load_error(spec, exc))
        raise


@app.post("/api/v1/requests/chat/completions", status_code=202)
def api_v1_enqueue_chat_completion(request: ApiChatCompletionRequest) -> dict:
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="The async queue endpoint only supports non-stream chat completions.",
        )

    prepared = prepare_api_chat_completion_request(request)
    spec = prepared["spec"]
    quantization_spec = prepared["quantization_spec"]
    request_id = request_queue.register_request(
        route="/api/v1/requests/chat/completions",
        runtime_family=quantization_spec.get("runtime_family"),
        model_key=spec["key"],
        quantization_key=quantization_spec["key"],
        stream=False,
        request_payload=build_request_payload_summary(
            route="/api/v1/requests/chat/completions",
            model_key=spec["key"],
            quantization_key=quantization_spec["key"],
            runtime_family=quantization_spec.get("runtime_family"),
            prompt=prepared["prompt"],
            system_prompt=prepared["system_prompt"],
            history_turns=len(prepared["history"]),
            max_new_tokens=max(32, min(int(request.max_tokens), 1024)),
            temperature=max(0.0, min(float(request.temperature), 2.0)),
            top_p=max(0.1, min(float(request.top_p), 1.0)),
            top_k=max(1, min(int(request.top_k), 128)),
            thinking=request.thinking,
            has_image=prepared["image"] is not None,
            has_audio=prepared["audio"] is not None,
            stream=False,
            extra={
                "tool_count": len(prepared["tools"]),
                "response_format": request.response_format.type
                if request.response_format
                else "text",
            },
        ),
        request_id=request.request_id,
    )

    runtime_messages = (
        prepared["runtime_messages"]
        if quantization_spec.get("runtime_family") == "transformers"
        else None
    )

    def run_background_chat_completion() -> None:
        try:
            request_queue.wait_for_turn(
                request_id,
                message="Generating async response.",
            )
            response_payload = service.generate(
                model_key=spec["key"],
                quantization_key=quantization_spec["key"],
                prompt=prepared["prompt"],
                system_prompt=prepared["system_prompt"],
                history=prepared["history"],
                image=prepared["image"],
                audio=prepared["audio"],
                max_new_tokens=max(32, min(int(request.max_tokens), 1024)),
                temperature=max(0.0, min(float(request.temperature), 2.0)),
                top_p=max(0.1, min(float(request.top_p), 1.0)),
                top_k=max(1, min(int(request.top_k), 128)),
                thinking=request.thinking,
                tools=prepared["tools"],
                override_messages=runtime_messages,
            )
            final_payload = build_chat_completion_response(
                request=request,
                prepared=prepared,
                response_payload=response_payload,
                request_id=request_id,
            )
            request_queue.finish(
                request_id,
                response_payload=final_payload,
                response_preview=summarize_prompt_text(
                    final_payload["choices"][0]["message"].get("content", ""),
                    limit=500,
                ),
            )
        except HTTPException as exc:
            request_queue.finish(request_id, error_text=str(exc.detail))
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception(
                "Async chat completion failed request_id=%s key=%s quantization=%s",
                request_id,
                spec["key"],
                quantization_spec["key"],
            )
            request_queue.finish(request_id, error_text=render_model_load_error(spec, exc))

    worker = threading.Thread(
        target=run_background_chat_completion,
        daemon=True,
        name=f"gemma-api-request-{request_id[:8]}",
    )
    worker.start()

    return {
        "accepted": True,
        "request_id": request_id,
        "status_url": f"/api/v1/requests/{request_id}",
        "queue": request_queue.snapshot(),
    }


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
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
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
    request_id = request_queue.register_request(
        route="/api/generate",
        runtime_family=quantization_spec.get("runtime_family"),
        model_key=spec["key"],
        quantization_key=quantization_spec["key"],
        stream=False,
        request_payload=build_request_payload_summary(
            route="/api/generate",
            model_key=spec["key"],
            quantization_key=quantization_spec["key"],
            runtime_family=quantization_spec.get("runtime_family"),
            prompt=normalized_prompt,
            system_prompt=normalized_system_prompt,
            history_turns=len(history),
            max_new_tokens=max(32, min(max_new_tokens, 1024)),
            temperature=max(0.0, min(temperature, 2.0)),
            top_p=max(0.1, min(top_p, 1.0)),
            top_k=max(1, min(top_k, 128)),
            thinking=thinking,
            has_image=image_payload is not None,
            has_audio=audio_payload is not None,
            stream=False,
            extra={"tts_enabled": tts_enabled},
        ),
    )
    logger.info(
        "Generate execution request_id=%s key=%s quantization=%s thinking=%s tts=%s prompt_chars=%s system_prompt_chars=%s history_turns=%s max_new_tokens=%s image=%s audio=%s",
        request_id,
        spec["key"],
        quantization_spec["key"],
        thinking,
        tts_enabled,
        len(normalized_prompt),
        len(normalized_system_prompt),
        len(history),
        max(32, min(max_new_tokens, 1024)),
        image_payload is not None,
        audio_payload is not None,
    )

    try:
        request_queue.wait_for_turn(request_id, message="Generating response.")
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

        response_payload["request_id"] = request_id
        response_payload["status_url"] = f"/api/v1/requests/{request_id}"
        request_queue.finish(
            request_id,
            response_payload=response_payload,
            response_preview=summarize_prompt_text(response_payload.get("reply", ""), limit=500),
        )
        return response_payload
    except HTTPException as exc:
        request_queue.finish(request_id, error_text=str(exc.detail))
        raise
    except Exception as exc:
        request_queue.finish(request_id, error_text=render_model_load_error(spec, exc))
        raise


@app.post("/api/generate-stream")
async def api_generate_stream(
    model_key: str = Form(DEFAULT_MODEL_KEY),
    quantization_key: str = Form(DEFAULT_QUANTIZATION_KEY),
    prompt: str = Form(""),
    system_prompt: str = Form(DEFAULT_SYSTEM_PROMPT),
    history_json: str = Form("[]"),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
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
    request_id = request_queue.register_request(
        route="/api/generate-stream",
        runtime_family=quantization_spec.get("runtime_family"),
        model_key=spec["key"],
        quantization_key=quantization_spec["key"],
        stream=True,
        request_payload=build_request_payload_summary(
            route="/api/generate-stream",
            model_key=spec["key"],
            quantization_key=quantization_spec["key"],
            runtime_family=quantization_spec.get("runtime_family"),
            prompt=normalized_prompt,
            system_prompt=normalized_system_prompt,
            history_turns=len(history),
            max_new_tokens=max(32, min(max_new_tokens, 1024)),
            temperature=max(0.0, min(temperature, 2.0)),
            top_p=max(0.1, min(top_p, 1.0)),
            top_k=max(1, min(top_k, 128)),
            thinking=thinking,
            has_image=image_payload is not None,
            has_audio=audio_payload is not None,
            stream=True,
            extra={"tts_enabled": tts_enabled},
        ),
    )
    logger.info(
        "Generate stream execution request_id=%s key=%s quantization=%s thinking=%s tts=%s prompt_chars=%s system_prompt_chars=%s history_turns=%s max_new_tokens=%s image=%s audio=%s",
        request_id,
        spec["key"],
        quantization_spec["key"],
        thinking,
        tts_enabled,
        len(normalized_prompt),
        len(normalized_system_prompt),
        len(history),
        max(32, min(max_new_tokens, 1024)),
        image_payload is not None,
        audio_payload is not None,
    )

    def event_stream():
        preview_text = ""
        try:
            request_queue.wait_for_turn(
                request_id,
                message="Generating streamed response.",
            )
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
                if event.get("event") == "start":
                    event["request_id"] = request_id
                    event["status_url"] = f"/api/v1/requests/{request_id}"
                elif event.get("event") == "token":
                    preview_text = summarize_prompt_text(
                        preview_text + event.get("text", ""),
                        limit=500,
                    )
                    request_queue.update_progress(
                        request_id,
                        status="running",
                        message="Streaming tokens.",
                        response_preview=preview_text,
                    )
                elif event.get("event") == "done":
                    event["payload"]["request_id"] = request_id
                    event["payload"]["status_url"] = f"/api/v1/requests/{request_id}"
                    request_queue.finish(
                        request_id,
                        response_payload=event["payload"],
                        response_preview=summarize_prompt_text(
                            event["payload"].get("reply", ""),
                            limit=500,
                        ),
                    )
                yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        except HTTPException as exc:
            request_queue.finish(request_id, error_text=str(exc.detail))
            yield (
                json.dumps(
                    {"event": "error", "request_id": request_id, "detail": str(exc.detail)},
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
            request_queue.finish(
                request_id,
                error_text=str(render_model_load_error(spec, exc)),
            )
            yield (
                json.dumps(
                    {
                        "event": "error",
                        "request_id": request_id,
                        "detail": str(render_model_load_error(spec, exc)),
                    },
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
