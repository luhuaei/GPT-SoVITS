from __future__ import annotations

import base64
from types import SimpleNamespace

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import api_v2


class DummyVoiceRegistry:
    def __init__(self):
        self.calls: list[str] = []
        self.registry = {
            "voice-1": {
                "ref_audio_path": "/runtime/voices/voice-1/reference.wav",
                "prompt_text": "参考文本",
                "prompt_lang": "zh",
            }
        }

    def get_voice(self, voice_id: str):
        self.calls.append(voice_id)
        voice = self.registry.get(voice_id)
        if voice is None:
            return None
        return {"voice_id": voice_id, **voice}

    def list_voices(self):
        return []

    def delete_voice(self, voice_id: str):
        return voice_id in self.registry

    def register_voice(self, source_path: str, filename: str, prompt_text: str, prompt_lang: str, name=None):
        return {
            "voice_id": "generated-voice",
            "name": name or "generated-voice",
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "ref_audio_path": source_path,
            "filename": filename,
        }


class FakeServiceState:
    last_instance: "FakeServiceState | None" = None

    def __init__(self, config_path: str, runtime_dir: str):
        self.config_path = config_path
        self.runtime_dir = runtime_dir
        self.model_id = api_v2.DEFAULT_MODEL_ID
        self.started_at = 0
        self.tts_config = SimpleNamespace(
            version="v2",
            languages={"auto", "zh", "en"},
            device="cpu",
            is_half=False,
        )
        self.tts_pipeline = object()
        self.voice_registry = DummyVoiceRegistry()
        self.request_gate = SimpleNamespace()
        FakeServiceState.last_instance = self


def build_client(monkeypatch):
    async def fake_run_tts_request(service, req, request_id, cleanup_files=None):
        return JSONResponse(
            {
                "request": req,
                "request_id": request_id,
                "cleanup_files": cleanup_files or [],
            }
        )

    monkeypatch.setattr(api_v2, "ServiceState", FakeServiceState)
    monkeypatch.setattr(api_v2, "run_tts_request", fake_run_tts_request)
    app = api_v2.create_app(config_path="test-config.yaml", runtime_dir="/tmp/gsv-runtime-test", argv=["pytest"])
    return TestClient(app)


def test_openai_audio_speech_accepts_direct_reference_metadata(monkeypatch):
    client = build_client(monkeypatch)
    monkeypatch.setattr(api_v2, "materialize_reference_audio", lambda runtime_dir, ref_audio: ("/tmp/direct.wav", ["/tmp/direct.wav"]))

    with client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": api_v2.DEFAULT_MODEL_ID,
                "input": "测试文本",
                "language": "zh",
                "ref_audio": "https://voice.example.com/reference.wav",
                "prompt_text": "参考文本",
                "prompt_lang": "zh",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request"]["ref_audio_path"] == "/tmp/direct.wav"
    assert payload["request"]["prompt_text"] == "参考文本"
    assert payload["request"]["prompt_lang"] == "zh"
    assert payload["cleanup_files"] == ["/tmp/direct.wav"]
    assert FakeServiceState.last_instance.voice_registry.calls == []


def test_openai_audio_speech_accepts_uploaded_reference_file(monkeypatch):
    client = build_client(monkeypatch)
    monkeypatch.setattr(
        api_v2,
        "save_temporary_uploaded_reference",
        lambda runtime_dir, upload: ("/tmp/uploaded-direct.wav", ["/tmp/uploaded-direct.wav"]),
    )

    with client:
        response = client.post(
            "/v1/audio/speech",
            data={
                "model": api_v2.DEFAULT_MODEL_ID,
                "input": "上传文件生成",
                "language": "zh",
                "prompt_text": "参考文本",
                "prompt_lang": "zh",
            },
            files={"ref_audio_file": ("reference.wav", b"fake-wave-data", "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request"]["ref_audio_path"] == "/tmp/uploaded-direct.wav"
    assert payload["request"]["prompt_text"] == "参考文本"
    assert payload["cleanup_files"] == ["/tmp/uploaded-direct.wav"]
    assert FakeServiceState.last_instance.voice_registry.calls == []


def test_openai_audio_speech_accepts_base64_reference(monkeypatch):
    client = build_client(monkeypatch)
    monkeypatch.setattr(
        api_v2,
        "decode_reference_audio_base64",
        lambda runtime_dir, ref_audio_base64: ("/tmp/base64-direct.wav", ["/tmp/base64-direct.wav"]),
    )

    with client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": api_v2.DEFAULT_MODEL_ID,
                "input": "base64 生成",
                "language": "zh",
                "ref_audio_base64": base64.b64encode(b"fake-wave-data").decode("ascii"),
                "prompt_text": "参考文本",
                "prompt_lang": "zh",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request"]["ref_audio_path"] == "/tmp/base64-direct.wav"
    assert payload["request"]["prompt_text"] == "参考文本"
    assert payload["cleanup_files"] == ["/tmp/base64-direct.wav"]
    assert FakeServiceState.last_instance.voice_registry.calls == []


def test_openai_audio_speech_keeps_legacy_voice_lookup(monkeypatch):
    client = build_client(monkeypatch)

    with client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": api_v2.DEFAULT_MODEL_ID,
                "input": "legacy voice request",
                "voice": "voice-1",
                "language": "zh",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request"]["ref_audio_path"] == "/runtime/voices/voice-1/reference.wav"
    assert payload["request"]["prompt_text"] == "参考文本"
    assert payload["request"]["prompt_lang"] == "zh"
    assert FakeServiceState.last_instance.voice_registry.calls == ["voice-1"]


def test_openai_audio_speech_rejects_missing_reference_source(monkeypatch):
    client = build_client(monkeypatch)

    with client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": api_v2.DEFAULT_MODEL_ID,
                "input": "missing voice metadata",
                "language": "zh",
            },
        )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "voice, reference_id, ref_audio, ref_audio_path, ref_audio_base64, or ref_audio_file is required"
    )


def test_create_voice_metadata_accepts_audio_url(monkeypatch):
    client = build_client(monkeypatch)

    with client:
        response = client.post(
            "/v1/voices/metadata",
            data={
                "audio_url": "https://voice.example.com/reference.wav",
                "prompt_text": "参考文本",
                "prompt_lang": "zh",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "voice_metadata"
    assert payload["source"] == "url"
    assert payload["ref_audio"] == "https://voice.example.com/reference.wav"
    assert payload["prompt_text"] == "参考文本"
    assert payload["prompt_lang"] == "zh"


def test_create_voice_metadata_accepts_file_upload(monkeypatch):
    client = build_client(monkeypatch)
    monkeypatch.setattr(api_v2, "save_uploaded_reference", lambda runtime_dir, upload: "/tmp/uploaded-reference.wav")

    with client:
        response = client.post(
            "/v1/voices/metadata",
            data={
                "prompt_text": "上传参考文本",
                "prompt_lang": "zh",
            },
            files={"file": ("reference.wav", b"fake-wave-data", "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "upload"
    assert payload["ref_audio"] == "/tmp/uploaded-reference.wav"
    assert payload["prompt_text"] == "上传参考文本"
    assert payload["prompt_lang"] == "zh"
