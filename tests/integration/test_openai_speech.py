from pathlib import Path

import pytest

from tests.integration.conftest import compare_text, synthesize_openai, transcribe_audio, write_json


@pytest.mark.integration
@pytest.mark.jetson
def test_openai_round_trip_zh(integration_config, registered_voice, healthcheck):
    assert healthcheck["status"] == "ok"
    payload = {
        "model": integration_config.model_id,
        "input": "今天的测试用于验证零样本音色克隆和中文流式语音合成是否可用。",
        "voice": registered_voice["voice_id"],
        "language": "zh",
        "response_format": "wav",
        "stream": True,
    }
    audio_path = integration_config.output_dir / "smoke" / "round_trip_zh.wav"
    synth = synthesize_openai(integration_config, payload, audio_path)
    transcript = transcribe_audio(integration_config, str(audio_path), "zh")
    comparison = compare_text(payload["input"], transcript, "zh")
    write_json(
        integration_config.output_dir / "smoke" / "round_trip_zh.json",
        {"synth": synth, "transcript": transcript, "comparison": comparison},
    )
    assert comparison["cer"] <= integration_config.zh_cer_threshold


@pytest.mark.integration
@pytest.mark.jetson
def test_openai_round_trip_en(integration_config, registered_voice):
    payload = {
        "model": integration_config.model_id,
        "input": "This test verifies that the OpenAI compatible speech endpoint can produce understandable English audio on Jetson hardware.",
        "voice": registered_voice["voice_id"],
        "language": "en",
        "response_format": "wav",
        "stream": False,
    }
    audio_path = integration_config.output_dir / "smoke" / "round_trip_en.wav"
    synth = synthesize_openai(integration_config, payload, audio_path)
    transcript = transcribe_audio(integration_config, str(audio_path), "en")
    comparison = compare_text(payload["input"], transcript, "en")
    write_json(
        integration_config.output_dir / "smoke" / "round_trip_en.json",
        {"synth": synth, "transcript": transcript, "comparison": comparison},
    )
    assert comparison["wer"] <= integration_config.en_wer_threshold

