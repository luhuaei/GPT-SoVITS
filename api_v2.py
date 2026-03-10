"""
HTTP APIs for GPT-SoVITS.

This module keeps the original `/tts` endpoints and adds:
- OpenAI-compatible `POST /v1/audio/speech`
- direct voice metadata preparation endpoint `POST /v1/voices/metadata`
- voice registration endpoints under `/v1/voices`
- service status endpoints `/healthz`, `/metrics`, `/v1/models`
"""

import argparse
import asyncio
import base64
import binascii
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import wave
from io import BytesIO
from pathlib import Path
from typing import Generator, Union
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
try:
    import onnxruntime
except Exception:
    onnxruntime = None

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(f"{now_dir}/GPT_SoVITS")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from gpt_sovits_api.request_gate import RequestGate
from gpt_sovits_api.voice_registry import VoiceRegistry
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()
cut_method_names = get_cut_method_names()

DEFAULT_CONFIG_PATH = os.environ.get("GPT_SOVITS_TTS_CONFIG", "GPT_SoVITS/configs/tts_infer.yaml")
DEFAULT_RUNTIME_DIR = os.environ.get("GPT_SOVITS_RUNTIME_DIR", "runtime")
DEFAULT_MODEL_ID = os.environ.get("GPT_SOVITS_MODEL_ID", "gpt-sovits-v2proplus-jetson")
DEFAULT_MAX_CONCURRENT = int(os.environ.get("GPT_SOVITS_MAX_CONCURRENT", "1"))
DEFAULT_MAX_QUEUE = int(os.environ.get("GPT_SOVITS_MAX_QUEUE", "16"))

OPENAI_SPEECH_OPENAPI_EXTRA = {
    "requestBody": {
        "content": {
            "application/json": {
                "examples": {
                    "legacy_voice_id": {
                        "summary": "Legacy voice ID mode",
                        "value": {
                            "model": DEFAULT_MODEL_ID,
                            "input": "这是一个旧模式示例。",
                            "voice": "voice-123",
                            "language": "zh",
                            "response_format": "wav",
                        },
                    },
                    "http_reference": {
                        "summary": "Direct HTTP reference audio",
                        "value": {
                            "model": DEFAULT_MODEL_ID,
                            "input": "这是一个 URL 参考音频示例。",
                            "language": "zh",
                            "ref_audio": "https://example.com/reference.wav",
                            "prompt_text": "今天天气很好，我们做一次测试。",
                            "prompt_lang": "zh",
                            "response_format": "wav",
                        },
                    },
                    "base64_reference": {
                        "summary": "Direct base64 reference audio",
                        "value": {
                            "model": DEFAULT_MODEL_ID,
                            "input": "这是一个 base64 参考音频示例。",
                            "language": "zh",
                            "ref_audio_base64": "UklGRiQAAABXQVZF...",
                            "prompt_text": "今天天气很好，我们做一次测试。",
                            "prompt_lang": "zh",
                            "response_format": "wav",
                        },
                    },
                }
            },
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string", "default": DEFAULT_MODEL_ID},
                        "input": {"type": "string"},
                        "voice": {"type": "string"},
                        "reference_id": {"type": "string"},
                        "ref_audio": {"type": "string"},
                        "ref_audio_path": {"type": "string"},
                        "ref_audio_base64": {"type": "string"},
                        "ref_audio_file": {"type": "string", "format": "binary"},
                        "prompt_text": {"type": "string"},
                        "prompt_lang": {"type": "string"},
                        "language": {"type": "string", "default": "auto"},
                        "response_format": {"type": "string", "default": "wav"},
                        "speed": {"type": "number", "default": 1.0},
                        "stream": {"type": "boolean", "default": False},
                        "text_split_method": {"type": "string", "default": "cut5"},
                        "batch_size": {"type": "integer", "default": 1},
                        "parallel_infer": {"type": "boolean", "default": True},
                        "temperature": {"type": "number", "default": 1.0},
                        "top_k": {"type": "integer", "default": 15},
                        "top_p": {"type": "number", "default": 1.0},
                        "repetition_penalty": {"type": "number", "default": 1.35},
                        "fragment_interval": {"type": "number", "default": 0.3},
                        "overlap_length": {"type": "integer", "default": 2},
                        "min_chunk_length": {"type": "integer", "default": 16},
                    },
                    "required": ["input"],
                }
            },
        }
    }
}


def configure_torch_runtime():
    torch_threads = int(os.environ.get("GPT_SOVITS_TORCH_NUM_THREADS", "1"))
    torch.set_num_threads(torch_threads)
    try:
        torch_interop_threads = int(os.environ.get("GPT_SOVITS_TORCH_INTEROP_THREADS", "1"))
        torch.set_num_interop_threads(torch_interop_threads)
    except RuntimeError:
        # PyTorch only allows setting inter-op threads once per process.
        pass


def describe_onnx_runtime() -> dict:
    if onnxruntime is None:
        return {"installed": False, "available_providers": []}
    try:
        available_providers = onnxruntime.get_available_providers()
    except Exception as exc:
        return {"installed": True, "available_providers": [], "error": str(exc)}
    return {"installed": True, "available_providers": available_providers}


class TTS_Request(BaseModel):
    text: str | None = None
    text_lang: str | None = None
    ref_audio_path: str | None = None
    aux_ref_audio_paths: list | None = None
    prompt_lang: str | None = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16


class OpenAISpeechRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL_ID, description="TTS model id. Must match `/v1/models`.")
    input: str = Field(description="Text to synthesize.")
    voice: str | None = Field(default=None, description="Legacy voice id stored in the local registry.")
    reference_id: str | None = Field(default=None, description="Alias of `voice` for compatibility.")
    ref_audio: str | None = Field(
        default=None,
        description="Direct reference audio input. Supports a local path or an HTTP/HTTPS URL.",
    )
    ref_audio_path: str | None = Field(
        default=None,
        description="Alias of `ref_audio`. Useful when the caller already stores the reference on shared storage.",
    )
    ref_audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded reference audio. Supports plain base64 or a data URL such as `data:audio/wav;base64,...`.",
    )
    prompt_text: str | None = Field(
        default=None,
        description="Transcript of the reference audio. Required when using `ref_audio`, `ref_audio_path`, `ref_audio_base64`, or `ref_audio_file`.",
    )
    prompt_lang: str | None = Field(
        default=None,
        description="Language of `prompt_text`. Required when using direct reference audio inputs.",
    )
    language: str = Field(default="auto", description="Target text language.")
    response_format: str = Field(default="wav", description="Output audio format: `wav`, `raw`, `ogg`, or `aac`.")
    speed: float = Field(default=1.0, description="Speech speed factor.")
    stream: bool = Field(default=False, description="Whether to stream audio chunks.")
    text_split_method: str = Field(default="cut5", description="Text segmentation method.")
    batch_size: int = Field(default=1, description="Inference batch size.")
    parallel_infer: bool = Field(default=True, description="Enable GPT-SoVITS parallel inference when applicable.")
    temperature: float = Field(default=1.0, description="Sampling temperature.")
    top_k: int = Field(default=15, description="Top-k sampling parameter.")
    top_p: float = Field(default=1.0, description="Top-p sampling parameter.")
    repetition_penalty: float = Field(default=1.35, description="Repetition penalty.")
    fragment_interval: float = Field(default=0.3, description="Streaming fragment interval.")
    overlap_length: int = Field(default=2, description="Streaming overlap length.")
    min_chunk_length: int = Field(default=16, description="Minimum streaming chunk length.")


class ServiceState:
    def __init__(self, config_path: str, runtime_dir: str):
        configure_torch_runtime()
        self.config_path = config_path
        self.runtime_dir = runtime_dir
        Path(runtime_dir).mkdir(parents=True, exist_ok=True)
        self.tts_config = TTS_Config(config_path)
        self.tts_pipeline = TTS(self.tts_config)
        self.voice_registry = VoiceRegistry(runtime_dir)
        self.request_gate = RequestGate(DEFAULT_MAX_CONCURRENT, DEFAULT_MAX_QUEUE)
        self.started_at = time.time()
        self.model_id = DEFAULT_MODEL_ID


def is_http_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def cleanup_paths(paths: list[str] | None):
    for path in paths or []:
        try:
            os.unlink(path)
        except FileNotFoundError:
            continue


def save_uploaded_reference(runtime_dir: str, upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix or ".wav"
    dest_dir = Path(runtime_dir) / "uploaded_references" / uuid.uuid4().hex
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"reference{suffix}"
    with dest_path.open("wb") as handle:
        handle.write(upload.file.read())
    return str(dest_path)


def save_temporary_uploaded_reference(runtime_dir: str, upload: UploadFile) -> tuple[str, list[str]]:
    suffix = Path(upload.filename or "").suffix or ".wav"
    temp_dir = Path(runtime_dir) / "speech_upload_references"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=False) as handle:
        handle.write(upload.file.read())
        temp_path = handle.name
    return temp_path, [temp_path]


def infer_suffix_from_base64_payload(ref_audio_base64: str) -> str:
    if ref_audio_base64.startswith("data:audio/"):
        mime = ref_audio_base64.split(";", 1)[0].split("/", 1)[1].lower()
        if mime in {"wav", "x-wav"}:
            return ".wav"
        if mime == "mpeg":
            return ".mp3"
        if mime in {"ogg", "aac", "flac"}:
            return f".{mime}"
    return ".wav"


def decode_reference_audio_base64(runtime_dir: str, ref_audio_base64: str) -> tuple[str, list[str]]:
    payload = ref_audio_base64.strip()
    suffix = infer_suffix_from_base64_payload(payload)
    if payload.startswith("data:"):
        try:
            payload = payload.split(",", 1)[1]
        except IndexError as exc:
            raise HTTPException(status_code=400, detail="invalid ref_audio_base64 data URL") from exc

    temp_dir = Path(runtime_dir) / "speech_upload_references"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        audio_bytes = base64.b64decode(payload, validate=True)
        with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=False) as handle:
            temp_path = handle.name
            handle.write(audio_bytes)
    except (binascii.Error, ValueError) as exc:
        cleanup_paths([temp_path] if temp_path else [])
        raise HTTPException(status_code=400, detail="invalid ref_audio_base64 payload") from exc
    return temp_path, [temp_path]


def download_reference_audio(runtime_dir: str, ref_audio: str) -> str:
    suffix = Path(urlparse(ref_audio).path).suffix or ".wav"
    temp_dir = Path(runtime_dir) / "remote_references"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=False) as handle:
            temp_path = handle.name
            with requests.get(ref_audio, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    except Exception:
        cleanup_paths([temp_path] if temp_path else [])
        raise
    return temp_path


def materialize_reference_audio(runtime_dir: str, ref_audio: str) -> tuple[str, list[str]]:
    if is_http_url(ref_audio):
        temp_path = download_reference_audio(runtime_dir, ref_audio)
        return temp_path, [temp_path]
    return ref_audio, []


async def parse_openai_speech_request(request: Request) -> tuple[OpenAISpeechRequest, UploadFile | None]:
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        payload = {}
        upload = None
        for key, value in form.items():
            if key == "ref_audio_file":
                upload = value
                continue
            payload[key] = value
        return OpenAISpeechRequest.model_validate(payload), upload

    payload = await request.json()
    return OpenAISpeechRequest.model_validate(payload), None


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    def handle_pack_ogg():
        with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)

    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as exc:
        print(f"RuntimeError: {exc}")
    except ValueError as exc:
        print(f"ValueError: {exc}")
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    del rate
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-vn",
            "-f",
            "adts",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str, argv: list[str]):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        raise SystemExit(0)


def get_service(app: FastAPI) -> ServiceState:
    service = getattr(app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="service is still starting")
    return service


def ensure_language_supported(service: ServiceState, value: str | None, field_name: str):
    if value in [None, ""]:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    normalized = value.lower()
    if normalized not in service.tts_config.languages:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name}: {value} is not supported in version {service.tts_config.version}",
        )
    return normalized


def validate_tts_request(service: ServiceState, req: dict):
    if req.get("ref_audio_path") in [None, ""]:
        raise HTTPException(status_code=400, detail="ref_audio_path is required")
    if req.get("text") in [None, ""]:
        raise HTTPException(status_code=400, detail="text is required")
    req["text_lang"] = ensure_language_supported(service, req.get("text_lang"), "text_lang")
    req["prompt_lang"] = ensure_language_supported(service, req.get("prompt_lang"), "prompt_lang")

    media_type = req.get("media_type", "wav")
    if media_type == "pcm":
        media_type = "raw"
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        raise HTTPException(status_code=400, detail=f"media_type: {media_type} is not supported")
    req["media_type"] = media_type

    text_split_method = req.get("text_split_method", "cut5")
    if text_split_method not in cut_method_names:
        raise HTTPException(status_code=400, detail=f"text_split_method:{text_split_method} is not supported")

    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    if streaming_mode == 0:
        streaming_mode = False
        return_fragment = False
        fixed_length_chunk = False
    elif streaming_mode is False:
        fixed_length_chunk = False
    elif streaming_mode == 1 or streaming_mode is True:
        streaming_mode = False
        return_fragment = True
        fixed_length_chunk = False
    elif streaming_mode == 2:
        streaming_mode = True
        return_fragment = False
        fixed_length_chunk = False
    elif streaming_mode == 3:
        streaming_mode = True
        return_fragment = False
        fixed_length_chunk = True
    else:
        raise HTTPException(
            status_code=400,
            detail="the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)",
        )

    req["streaming_mode"] = streaming_mode
    req["return_fragment"] = return_fragment
    req["fixed_length_chunk"] = fixed_length_chunk
    return req


def release_ticket_from_thread(ticket, loop: asyncio.AbstractEventLoop):
    future = asyncio.run_coroutine_threadsafe(ticket.release(), loop)
    future.result(timeout=10)


async def run_tts_request(service: ServiceState, req: dict, request_id: str, cleanup_files: list[str] | None = None):
    try:
        validated_req = validate_tts_request(service, req)
    except Exception:
        cleanup_paths(cleanup_files)
        raise
    try:
        ticket = await service.request_gate.acquire(request_id)
    except RuntimeError as exc:
        cleanup_paths(cleanup_files)
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    media_type = validated_req["media_type"]
    streaming_enabled = validated_req["streaming_mode"] or validated_req["return_fragment"]
    loop = asyncio.get_running_loop()

    try:
        tts_generator = service.tts_pipeline.run(validated_req)
    except Exception:
        await ticket.release()
        cleanup_paths(cleanup_files)
        raise

    if streaming_enabled:
        def streaming_generator():
            first_chunk = True
            try:
                for sr, chunk in tts_generator:
                    if first_chunk and media_type == "wav":
                        yield wave_header_chunk(sample_rate=sr)
                        first_chunk = False
                        active_media_type = "raw"
                    else:
                        active_media_type = media_type
                    yield pack_audio(BytesIO(), chunk, sr, active_media_type).getvalue()
            finally:
                try:
                    release_ticket_from_thread(ticket, loop)
                finally:
                    cleanup_paths(cleanup_files)

        return StreamingResponse(streaming_generator(), media_type=f"audio/{media_type}")

    try:
        sr, audio_data = next(tts_generator)
        payload = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
        return Response(payload, media_type=f"audio/{media_type}")
    finally:
        try:
            await ticket.release()
        finally:
            cleanup_paths(cleanup_files)


def resolve_openai_reference(
    service: ServiceState,
    body: OpenAISpeechRequest,
    ref_audio_file: UploadFile | None = None,
) -> tuple[dict, list[str]]:
    if ref_audio_file is not None:
        prompt_text = (body.prompt_text or "").strip()
        if not prompt_text:
            raise HTTPException(status_code=400, detail="prompt_text is required when ref_audio is provided")
        normalized_prompt_lang = ensure_language_supported(service, body.prompt_lang, "prompt_lang")
        ref_audio_path, cleanup_files = save_temporary_uploaded_reference(service.runtime_dir, ref_audio_file)
        return {
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": [],
            "prompt_text": prompt_text,
            "prompt_lang": normalized_prompt_lang,
        }, cleanup_files

    if body.ref_audio_base64:
        prompt_text = (body.prompt_text or "").strip()
        if not prompt_text:
            raise HTTPException(status_code=400, detail="prompt_text is required when ref_audio is provided")
        normalized_prompt_lang = ensure_language_supported(service, body.prompt_lang, "prompt_lang")
        ref_audio_path, cleanup_files = decode_reference_audio_base64(service.runtime_dir, body.ref_audio_base64)
        return {
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": [],
            "prompt_text": prompt_text,
            "prompt_lang": normalized_prompt_lang,
        }, cleanup_files

    ref_audio = body.ref_audio or body.ref_audio_path
    if ref_audio:
        prompt_text = (body.prompt_text or "").strip()
        if not prompt_text:
            raise HTTPException(status_code=400, detail="prompt_text is required when ref_audio is provided")
        normalized_prompt_lang = ensure_language_supported(service, body.prompt_lang, "prompt_lang")
        ref_audio_path, cleanup_files = materialize_reference_audio(service.runtime_dir, ref_audio)
        return {
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": [],
            "prompt_text": prompt_text,
            "prompt_lang": normalized_prompt_lang,
        }, cleanup_files

    voice_id = body.reference_id or body.voice
    if not voice_id:
        raise HTTPException(
            status_code=400,
            detail="voice, reference_id, ref_audio, ref_audio_path, ref_audio_base64, or ref_audio_file is required",
        )
    voice = service.voice_registry.get_voice(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice not found: {voice_id}")
    return {
        "ref_audio_path": voice["ref_audio_path"],
        "aux_ref_audio_paths": [],
        "prompt_text": voice["prompt_text"],
        "prompt_lang": voice["prompt_lang"],
    }, []


def build_openai_request(
    service: ServiceState,
    body: OpenAISpeechRequest,
    ref_audio_file: UploadFile | None = None,
) -> tuple[dict, list[str]]:
    if body.model != service.model_id:
        raise HTTPException(status_code=400, detail=f"unsupported model: {body.model}")

    reference_inputs, cleanup_files = resolve_openai_reference(service, body, ref_audio_file=ref_audio_file)

    return {
        "text": body.input,
        "text_lang": body.language,
        "ref_audio_path": reference_inputs["ref_audio_path"],
        "aux_ref_audio_paths": reference_inputs["aux_ref_audio_paths"],
        "prompt_text": reference_inputs["prompt_text"],
        "prompt_lang": reference_inputs["prompt_lang"],
        "top_k": body.top_k,
        "top_p": body.top_p,
        "temperature": body.temperature,
        "text_split_method": body.text_split_method,
        "batch_size": body.batch_size,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": body.speed,
        "fragment_interval": body.fragment_interval,
        "seed": -1,
        "media_type": body.response_format,
        "streaming_mode": 2 if body.stream else False,
        "parallel_infer": body.parallel_infer,
        "repetition_penalty": body.repetition_penalty,
        "sample_steps": 32,
        "super_sampling": False,
        "overlap_length": body.overlap_length,
        "min_chunk_length": body.min_chunk_length,
    }, cleanup_files


def create_app(config_path: str = DEFAULT_CONFIG_PATH, runtime_dir: str = DEFAULT_RUNTIME_DIR, argv: list[str] | None = None):
    app = FastAPI(title="GPT-SoVITS API")
    app.state.service = None
    app.state.argv = list(argv or sys.argv)

    @app.on_event("startup")
    async def startup_event():
        app.state.service = ServiceState(config_path, runtime_dir)

    @app.get("/healthz")
    async def healthz():
        service = get_service(app)
        gate = await service.request_gate.snapshot()
        return {
            "status": "ok",
            "version": service.tts_config.version,
            "model_id": service.model_id,
            "started_at": service.started_at,
            "uptime_seconds": max(0.0, time.time() - service.started_at),
            "voice_count": len(service.voice_registry.registry),
            "queue": gate,
            "config": {
                "device": str(service.tts_config.device),
                "is_half": service.tts_config.is_half,
                "config_path": service.config_path,
                "g2pw_enabled": os.environ.get("GPT_SOVITS_ENABLE_G2PW", "auto"),
                "require_onnx_gpu": os.environ.get("GPT_SOVITS_REQUIRE_ONNX_GPU", "false"),
            },
            "onnxruntime": describe_onnx_runtime(),
        }

    @app.get("/metrics")
    async def metrics():
        service = get_service(app)
        gate = await service.request_gate.snapshot()
        return {
            "model_id": service.model_id,
            "voice_count": len(service.voice_registry.registry),
            "queue": gate,
            "onnxruntime": describe_onnx_runtime(),
        }

    @app.get("/v1/models")
    async def list_models():
        service = get_service(app)
        return {
            "object": "list",
            "data": [
                {
                    "id": service.model_id,
                    "object": "model",
                    "created": int(service.started_at),
                    "owned_by": "gpt-sovits",
                }
            ],
        }

    @app.post(
        "/v1/audio/speech",
        summary="OpenAI-compatible speech synthesis",
        description=(
            "Supports both legacy `voice` / `reference_id` mode and direct reference-audio mode. "
            "Direct mode accepts `ref_audio` or `ref_audio_path` as a local path or HTTP/HTTPS URL, "
            "`ref_audio_base64` in JSON, or `ref_audio_file` in multipart form-data. "
            "When using direct reference audio, `prompt_text` and `prompt_lang` are required."
        ),
        openapi_extra=OPENAI_SPEECH_OPENAPI_EXTRA,
    )
    async def openai_audio_speech(request: Request):
        service = get_service(app)
        body, ref_audio_file = await parse_openai_speech_request(request)
        req, cleanup_files = build_openai_request(service, body, ref_audio_file=ref_audio_file)
        request_id = f"speech-{uuid.uuid4().hex}"
        try:
            return await run_tts_request(service, req, request_id, cleanup_files=cleanup_files)
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse(status_code=400, content={"message": "tts failed", "exception": str(exc)})

    @app.post(
        "/v1/voices/metadata",
        summary="Prepare TTS voice metadata without registry persistence",
        description=(
            "Accepts either an uploaded reference file or an `audio_url`, plus `prompt_text` and `prompt_lang`, "
            "and returns the direct metadata fields required by `/v1/audio/speech`. "
            "This endpoint does not write to the local voice registry."
        ),
    )
    async def create_voice_metadata(
        file: UploadFile | None = File(default=None),
        audio_url: str | None = Form(default=None),
        prompt_text: str = Form(...),
        prompt_lang: str = Form(...),
    ):
        service = get_service(app)
        normalized_prompt_lang = ensure_language_supported(service, prompt_lang, "prompt_lang")
        normalized_prompt_text = prompt_text.strip()
        if not normalized_prompt_text:
            raise HTTPException(status_code=400, detail="prompt_text is required")
        if (file is None and not audio_url) or (file is not None and audio_url):
            raise HTTPException(status_code=400, detail="exactly one of file or audio_url is required")

        if file is not None:
            ref_audio = save_uploaded_reference(service.runtime_dir, file)
            source = "upload"
        else:
            if not is_http_url(audio_url):
                raise HTTPException(status_code=400, detail="audio_url must be a valid http or https URL")
            ref_audio = audio_url
            source = "url"

        return {
            "object": "voice_metadata",
            "source": source,
            "ref_audio": ref_audio,
            "ref_audio_path": ref_audio,
            "prompt_text": normalized_prompt_text,
            "prompt_lang": normalized_prompt_lang,
        }

    @app.post("/v1/voices")
    async def create_voice(
        file: UploadFile = File(...),
        prompt_text: str = Form(...),
        prompt_lang: str = Form(...),
        name: str | None = Form(default=None),
    ):
        service = get_service(app)
        normalized_prompt_lang = ensure_language_supported(service, prompt_lang, "prompt_lang")
        if not prompt_text.strip():
            raise HTTPException(status_code=400, detail="prompt_text is required")
        suffix = Path(file.filename or "").suffix or ".wav"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                temp_path = handle.name
                handle.write(await file.read())
            voice = service.voice_registry.register_voice(
                source_path=temp_path,
                filename=file.filename or Path(temp_path).name,
                prompt_text=prompt_text.strip(),
                prompt_lang=normalized_prompt_lang,
                name=name,
            )
            return voice
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    @app.get("/v1/voices")
    async def list_voices():
        service = get_service(app)
        return {"object": "list", "data": service.voice_registry.list_voices()}

    @app.get("/v1/voices/{voice_id}")
    async def get_voice(voice_id: str):
        service = get_service(app)
        voice = service.voice_registry.get_voice(voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"voice not found: {voice_id}")
        return voice

    @app.delete("/v1/voices/{voice_id}")
    async def delete_voice(voice_id: str):
        service = get_service(app)
        if not service.voice_registry.delete_voice(voice_id):
            raise HTTPException(status_code=404, detail=f"voice not found: {voice_id}")
        return {"id": voice_id, "deleted": True}

    @app.get("/control")
    async def control(command: str | None = None):
        if command is None:
            return JSONResponse(status_code=400, content={"message": "command is required"})
        handle_control(command, app.state.argv)
        return JSONResponse(status_code=200, content={"message": "success"})

    @app.get("/tts")
    async def tts_get_endpoint(
        text: str | None = None,
        text_lang: str | None = None,
        ref_audio_path: str | None = None,
        aux_ref_audio_paths: list | None = None,
        prompt_lang: str | None = None,
        prompt_text: str = "",
        top_k: int = 15,
        top_p: float = 1,
        temperature: float = 1,
        text_split_method: str = "cut5",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        seed: int = -1,
        media_type: str = "wav",
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35,
        sample_steps: int = 32,
        super_sampling: bool = False,
        streaming_mode: Union[bool, int] = False,
        overlap_length: int = 2,
        min_chunk_length: int = 16,
    ):
        service = get_service(app)
        req = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": int(batch_size),
            "batch_threshold": float(batch_threshold),
            "speed_factor": float(speed_factor),
            "split_bucket": split_bucket,
            "fragment_interval": fragment_interval,
            "seed": seed,
            "media_type": media_type,
            "streaming_mode": streaming_mode,
            "parallel_infer": parallel_infer,
            "repetition_penalty": float(repetition_penalty),
            "sample_steps": int(sample_steps),
            "super_sampling": super_sampling,
            "overlap_length": int(overlap_length),
            "min_chunk_length": int(min_chunk_length),
        }
        try:
            return await run_tts_request(service, req, f"tts-{uuid.uuid4().hex}")
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse(status_code=400, content={"message": "tts failed", "exception": str(exc)})

    @app.post("/tts")
    async def tts_post_endpoint(request: TTS_Request):
        service = get_service(app)
        req = request.model_dump()
        try:
            return await run_tts_request(service, req, f"tts-{uuid.uuid4().hex}")
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse(status_code=400, content={"message": "tts failed", "exception": str(exc)})

    @app.get("/set_refer_audio")
    async def set_refer_audio(refer_audio_path: str | None = None):
        service = get_service(app)
        try:
            service.tts_pipeline.set_ref_audio(refer_audio_path)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"message": "set refer audio failed", "exception": str(exc)})
        return JSONResponse(status_code=200, content={"message": "success"})

    @app.get("/set_gpt_weights")
    async def set_gpt_weights(weights_path: str | None = None):
        service = get_service(app)
        try:
            if weights_path in ["", None]:
                return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
            service.tts_pipeline.init_t2s_weights(weights_path)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "exception": str(exc)})
        return JSONResponse(status_code=200, content={"message": "success"})

    @app.get("/set_sovits_weights")
    async def set_sovits_weights(weights_path: str | None = None):
        service = get_service(app)
        try:
            if weights_path in ["", None]:
                return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
            service.tts_pipeline.init_vits_weights(weights_path)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "exception": str(exc)})
        return JSONResponse(status_code=200, content={"message": "success"})

    return app


APP = create_app()


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-SoVITS api")
    parser.add_argument("-c", "--tts_config", type=str, default=DEFAULT_CONFIG_PATH, help="tts_infer path")
    parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help='default: "127.0.0.1"')
    parser.add_argument("-p", "--port", type=int, default=9880, help="default: 9880")
    parser.add_argument("--runtime-dir", type=str, default=DEFAULT_RUNTIME_DIR, help="runtime directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    host = None if args.bind_addr == "None" else args.bind_addr
    app = create_app(config_path=args.tts_config, runtime_dir=args.runtime_dir, argv=sys.argv)
    try:
        uvicorn.run(app=app, host=host, port=args.port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        raise SystemExit(1)
