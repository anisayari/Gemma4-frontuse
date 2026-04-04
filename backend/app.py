from __future__ import annotations

import ctypes
import gc
import io
import json
import logging
import math
import os
import platform
import re
import threading
import time
import uuid
import wave
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

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
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForMultimodalLM, AutoProcessor


BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "web" / "dist"
CACHE_DIR = Path(os.getenv("HF_HOME", BASE_DIR / ".hf-cache"))
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "gemma4-lab.log"
TTS_DIR = BASE_DIR / "tts"
TTS_VOICE_DIR = TTS_DIR / "voices"
TTS_GENERATED_DIR = TTS_DIR / "generated"
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
    "(updated 2026-04-02). This local backend currently runs the official Hugging Face "
    "checkpoints in BF16 only; SFP8 and Q4_0 are shown for planning and UI comparison, "
    "but require a different runtime or checkpoint format."
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
        "doc_summary": "Largest dense Gemma 4 variant for local workstation use.",
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
        "doc_summary": "Default 16-bit Hugging Face path used by this local backend.",
    },
    {
        "key": "sfp8",
        "label": "SFP8",
        "precision_bits": 8,
        "runtime_supported": False,
        "status": "planning-only",
        "doc_summary": "Official Google memory estimate exposed for planning. Not loadable in this Transformers backend as-is.",
    },
    {
        "key": "q4_0",
        "label": "Q4_0",
        "precision_bits": 4,
        "runtime_supported": False,
        "status": "planning-only",
        "doc_summary": "Official Google memory estimate exposed for planning. Requires a quantized runtime/checkpoint family outside this backend.",
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
        "doc_summary": spec["doc_summary"],
    }


def serialize_quantization_spec(spec: dict) -> dict:
    return {
        "key": spec["key"],
        "label": spec["label"],
        "precision_bits": spec["precision_bits"],
        "runtime_supported": spec["runtime_supported"],
        "status": spec["status"],
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

    memory_estimate = model_spec["memory_requirements_gib"].get(quantization_spec["key"])
    if memory_estimate is None:
        return (
            f"{quantization_spec['label']} is not available for {model_spec['label']} in the "
            "official Gemma 4 memory table."
        )

    return (
        f"{model_spec['label']} with {quantization_spec['label']} is exposed in the UI using "
        f"Google's memory estimate ({memory_estimate:.1f} GiB), but this local backend cannot "
        "load it yet. The current FastAPI + Hugging Face Transformers path uses the official "
        "BF16 checkpoints. SFP8/Q4_0 need a different runtime or quantized checkpoint format."
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

    return f"{spec['label']} failed to load: {raw_message}"


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


class GemmaService:
    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._current_model_key = None
        self._current_quantization_key = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self.loaded_at = None

    @property
    def is_loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def _unload_current_model(self) -> None:
        self._processor = None
        self._model = None
        self._current_model_key = None
        self._current_quantization_key = None
        self.loaded_at = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_loaded(
        self, model_key: str | None, quantization_key: str | None = None
    ) -> tuple[object, object, dict, dict]:
        spec = get_model_spec(model_key)
        quantization_spec = get_quantization_spec(quantization_key)
        requested_key = spec["key"]
        requested_quantization_key = quantization_spec["key"]

        if (
            self.is_loaded
            and self._current_model_key == requested_key
            and self._current_quantization_key == requested_quantization_key
        ):
            return self._processor, self._model, spec, quantization_spec

        with self._load_lock:
            if (
                self.is_loaded
                and self._current_model_key == requested_key
                and self._current_quantization_key == requested_quantization_key
            ):
                return self._processor, self._model, spec, quantization_spec

            quantization_error = preflight_quantization_support(spec, quantization_spec)
            if quantization_error is not None:
                logger.warning(
                    "Model load blocked key=%s quantization=%s detail=%s",
                    spec["key"],
                    quantization_spec["key"],
                    quantization_error,
                )
                raise HTTPException(status_code=501, detail=quantization_error)

            preflight_error = preflight_model_load(spec)
            if preflight_error is not None:
                self._unload_current_model()
                logger.warning(
                    "Model load blocked key=%s quantization=%s hf_model_id=%s detail=%s",
                    spec["key"],
                    quantization_spec["key"],
                    spec["hf_model_id"],
                    preflight_error,
                )
                raise HTTPException(status_code=503, detail=preflight_error)

            if self.is_loaded and (
                self._current_model_key != requested_key
                or self._current_quantization_key != requested_quantization_key
            ):
                self._unload_current_model()

            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self._processor = AutoProcessor.from_pretrained(
                    spec["hf_model_id"],
                    cache_dir=str(CACHE_DIR),
                    local_files_only=LOCAL_FILES_ONLY,
                )
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    spec["hf_model_id"],
                    cache_dir=str(CACHE_DIR),
                    local_files_only=LOCAL_FILES_ONLY,
                    dtype=torch.bfloat16,
                    device_map="auto",
                )
                self._current_model_key = requested_key
                self._current_quantization_key = requested_quantization_key
                self.loaded_at = time.time()
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

    def health(self) -> dict:
        active_spec = get_model_spec(self._current_model_key or DEFAULT_MODEL_KEY)
        active_quantization = get_quantization_spec(
            self._current_quantization_key or DEFAULT_QUANTIZATION_KEY
        )
        windows_commit_snapshot = get_windows_commit_snapshot()
        gpu_total_memory_gib = get_gpu_total_memory_gib()
        return {
            "status": "ready",
            "active_model_key": active_spec["key"],
            "active_model": serialize_model_spec(active_spec),
            "active_quantization_key": active_quantization["key"],
            "active_quantization": serialize_quantization_spec(active_quantization),
            "tts": tts_service.health(),
            "loaded": self.is_loaded,
            "local_files_only": LOCAL_FILES_ONLY,
            "cache_dir": str(CACHE_DIR),
            "log_path": str(LOG_PATH),
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
        processor, model, spec, quantization_spec = self.ensure_loaded(
            model_key, quantization_key
        )

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

            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_kwargs)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
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


@app.post("/api/models/load")
def api_load_model(request: ModelLoadRequest) -> dict:
    logger.info(
        "Model load requested key=%s quantization=%s",
        request.model_key,
        request.quantization_key,
    )
    _, _, spec, quantization_spec = service.ensure_loaded(
        request.model_key, request.quantization_key
    )
    return {
        "message": f"{spec['label']} is loaded in {quantization_spec['label']}.",
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
    spec = get_model_spec(model_key)
    quantization_spec = get_quantization_spec(quantization_key)
    logger.info(
        "Generate requested key=%s quantization=%s image=%s audio=%s thinking=%s tts=%s",
        spec["key"],
        quantization_spec["key"],
        image is not None,
        audio is not None,
        thinking,
        tts_enabled,
    )
    history = decode_history(history_json)
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

    processor, _, _, _ = service.ensure_loaded(spec["key"], quantization_spec["key"])
    audio_payload = load_audio(audio, processor.feature_extractor.sampling_rate)

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

    response_payload = service.generate(
        model_key=spec["key"],
        quantization_key=quantization_spec["key"],
        prompt=normalized_prompt,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        history=history[-8:],
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
