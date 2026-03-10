import pytest

from tests.integration.conftest import compare_text, synthesize_openai, transcribe_audio, write_json


@pytest.mark.integration
@pytest.mark.jetson
@pytest.mark.slow
@pytest.mark.parametrize("language,stream", [("zh", False), ("en", False)])
def test_longform_generation_and_validation(integration_config, registered_voice, longform_texts, language, stream):
    text = longform_texts[language]
    payload = {
        "model": integration_config.model_id,
        "input": text,
        "voice": registered_voice["voice_id"],
        "language": language,
        "response_format": "wav",
        "stream": stream,
        "batch_size": 4,
    }
    suffix = "stream" if stream else "full"
    audio_path = integration_config.output_dir / "longform" / f"{language}_{suffix}.wav"
    synth = synthesize_openai(integration_config, payload, audio_path)
    transcript = transcribe_audio(integration_config, str(audio_path), language)
    comparison = compare_text(text, transcript, language)
    write_json(
        integration_config.output_dir / "longform" / f"{language}_{suffix}.json",
        {"synth": synth, "transcript": transcript, "comparison": comparison},
    )
    if language == "zh":
        assert comparison["cer"] <= integration_config.zh_cer_threshold
    else:
        assert comparison["wer"] <= integration_config.en_wer_threshold
