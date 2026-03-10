import json
import os
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pytest
import requests

from gpt_sovits_api.text_tools import (
    build_longform_chinese_text,
    build_longform_english_text,
    cer,
    normalize_asr_text,
    wer,
)


@dataclass
class IntegrationConfig:
    tts_base_url: str
    asr_base_url: str
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str
    model_id: str
    output_dir: Path
    zh_cer_threshold: float
    en_wer_threshold: float
    repeats: int
    stream_sample_rate: int


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False, help="run integration tests")


def require_integration(pytestconfig):
    if not pytestconfig.getoption("--run-integration"):
        pytest.skip("integration tests require --run-integration")


@pytest.fixture(scope="session")
def integration_config(pytestconfig):
    require_integration(pytestconfig)
    ref_audio_path = os.environ.get("TEST_REF_AUDIO_PATH")
    prompt_text = os.environ.get("TEST_PROMPT_TEXT")
    if not ref_audio_path or not prompt_text:
        pytest.skip("TEST_REF_AUDIO_PATH and TEST_PROMPT_TEXT are required")
    if not Path(ref_audio_path).exists():
        pytest.skip(f"reference audio not found: {ref_audio_path}")

    output_dir = Path(os.environ.get("TEST_OUTPUT_DIR", "test-results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return IntegrationConfig(
        tts_base_url=os.environ.get("TTS_BASE_URL", "http://127.0.0.1:9880"),
        asr_base_url=os.environ.get("ASR_BASE_URL", "http://192.168.1.230:10001"),
        ref_audio_path=ref_audio_path,
        prompt_text=prompt_text,
        prompt_lang=os.environ.get("TEST_PROMPT_LANG", "zh"),
        model_id=os.environ.get("TEST_MODEL_ID", "gpt-sovits-v2proplus-jetson"),
        output_dir=output_dir,
        zh_cer_threshold=float(os.environ.get("TEST_ZH_CER_THRESHOLD", "0.18")),
        en_wer_threshold=float(os.environ.get("TEST_EN_WER_THRESHOLD", "0.25")),
        repeats=int(os.environ.get("TEST_CONCURRENCY_REPEATS", "1")),
        stream_sample_rate=int(os.environ.get("TEST_STREAM_SAMPLE_RATE", "32000")),
    )


@pytest.fixture(scope="session")
def healthcheck(integration_config):
    response = requests.get(f"{integration_config.tts_base_url}/healthz", timeout=30)
    response.raise_for_status()
    return response.json()


@pytest.fixture(scope="session")
def registered_voice(integration_config):
    with open(integration_config.ref_audio_path, "rb") as handle:
        response = requests.post(
            f"{integration_config.tts_base_url}/v1/voices",
            files={"file": (Path(integration_config.ref_audio_path).name, handle, "audio/wav")},
            data={
                "prompt_text": integration_config.prompt_text,
                "prompt_lang": integration_config.prompt_lang,
                "name": "integration-test-voice",
            },
            timeout=120,
        )
    response.raise_for_status()
    voice = response.json()
    yield voice
    requests.delete(f"{integration_config.tts_base_url}/v1/voices/{voice['voice_id']}", timeout=30)


def synthesize_openai(integration_config, payload, output_path: Path):
    started_at = time.perf_counter()
    response = requests.post(
        f"{integration_config.tts_base_url}/v1/audio/speech",
        json=payload,
        stream=payload.get("stream", False),
        timeout=(30, 900),
    )
    response.raise_for_status()

    first_chunk_seconds = None
    content = bytearray()
    if payload.get("stream", False):
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - started_at
            content.extend(chunk)
    else:
        content.extend(response.content)

    total_seconds = time.perf_counter() - started_at
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_bytes = bytes(content)
    if payload.get("stream", False) and payload.get("response_format") == "wav":
        audio_bytes = rebuild_streamed_wav(audio_bytes, integration_config.stream_sample_rate)
    output_path.write_bytes(audio_bytes)
    return {
        "audio_path": str(output_path),
        "bytes": len(audio_bytes),
        "status_code": response.status_code,
        "first_chunk_seconds": first_chunk_seconds,
        "total_seconds": total_seconds,
    }


def rebuild_streamed_wav(content: bytes, sample_rate: int) -> bytes:
    pcm_bytes = content[44:] if content.startswith(b"RIFF") and len(content) >= 44 else content
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_buffer.getvalue()


def transcribe_audio(integration_config, audio_path: str, language: str | None):
    with open(audio_path, "rb") as handle:
        response = requests.post(
            f"{integration_config.asr_base_url}/v1/audio/transcriptions",
            files={"file": (Path(audio_path).name, handle, "audio/wav")},
            data={
                "model": "sensevoice-small",
                "response_format": "json",
                "language": language or "",
                "enable_punctuation": "true",
                "enable_itn": "true",
            },
            timeout=(30, 900),
        )
    response.raise_for_status()
    return response.json()["text"]


def compare_text(expected: str, actual: str, language: str):
    normalized_expected = normalize_asr_text(expected, language=language)
    normalized_actual = normalize_asr_text(actual, language=language)
    if language == "en":
        score = wer(normalized_expected, normalized_actual)
        score_name = "wer"
    else:
        score = cer(normalized_expected, normalized_actual)
        score_name = "cer"
    return {
        "expected": normalized_expected,
        "actual": normalized_actual,
        score_name: score,
        "score_name": score_name,
    }


def write_json(output_path: Path, payload: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture(scope="session")
def longform_texts():
    return {
        "zh": build_longform_chinese_text(),
        "en": build_longform_english_text(),
    }
