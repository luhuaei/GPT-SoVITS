#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import wave
from contextlib import closing
from pathlib import Path
from urllib import request


DEFAULT_CANDIDATES = [
    "/home/nvidia/qwen3-tts-compose-candidate-results/jetson-agx-orin/remote_vllm/compose-opt-compose_graph-smoke-20260308/regression.wav",
    "/home/nvidia/qwen3-tts-compose-candidate-results/jetson-agx-orin/remote_vllm/compose-opt-compose_graph_async-smoke-20260308/stream_regression.wav",
    "/home/nvidia/MeloTTS-OV/speech.mp3",
]


def audio_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    with closing(wave.open(str(path), "rb")) as handle:
        return handle.getnframes() / handle.getframerate()


def choose_reference_path() -> Path:
    explicit = os.environ.get("TEST_REF_AUDIO_PATH")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"reference audio not found: {path}")
        return path
    for candidate in DEFAULT_CANDIDATES:
        path = Path(candidate)
        if not path.exists():
            continue
        duration = audio_duration(path)
        if duration is None or 3.0 <= duration <= 10.0:
            return path
    raise FileNotFoundError("no suitable reference audio found on Jetson host")


def asr_prompt_text(audio_path: Path) -> str:
    asr_base = os.environ.get("ASR_BASE_URL", "http://192.168.1.230:10001")
    boundary = "----gsvitsrefboundary"
    with audio_path.open("rb") as handle:
        file_bytes = handle.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + file_bytes + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        "sensevoice-small\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="response_format"\r\n\r\n'
        "json\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    req = request.Request(
        f"{asr_base}/v1/audio/transcriptions",
        method="POST",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with request.urlopen(req, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = payload["text"]
    text = subprocess.run(
        [
            sys.executable,
            "-c",
            "from gpt_sovits_api.text_tools import strip_sensevoice_tags; import sys; print(strip_sensevoice_tags(sys.stdin.read()).strip())",
        ],
        input=text,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if not text:
        raise RuntimeError("ASR returned empty prompt text for reference audio")
    return text


def main():
    ref_audio = choose_reference_path()
    prompt_text = os.environ.get("TEST_PROMPT_TEXT") or asr_prompt_text(ref_audio)
    prompt_lang = os.environ.get("TEST_PROMPT_LANG", "zh")
    result = {
        "ref_audio_path": str(ref_audio),
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
