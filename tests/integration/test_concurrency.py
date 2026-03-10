import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from tests.integration.conftest import compare_text, synthesize_openai, transcribe_audio, write_json


def run_single_request(integration_config, voice_id: str, concurrency_level: int, item_index: int):
    text = f"并发{concurrency_level}，请求{item_index}。验证流式语音。"
    audio_path = integration_config.output_dir / "concurrency" / f"c{concurrency_level}" / f"request_{item_index}.wav"
    payload = {
        "model": integration_config.model_id,
        "input": text,
        "voice": voice_id,
        "language": "zh",
        "response_format": "wav",
        "stream": True,
        "text_split_method": "cut0",
    }
    try:
        synth = synthesize_openai(integration_config, payload, audio_path)
        transcript = transcribe_audio(integration_config, str(audio_path), "zh")
        comparison = compare_text(text, transcript, "zh")
        return {
            "payload": payload,
            "ok": True,
            "synth": synth,
            "transcript": transcript,
            "comparison": comparison,
        }
    except Exception as exc:
        return {
            "payload": payload,
            "ok": False,
            "error": str(exc),
        }


@pytest.mark.integration
@pytest.mark.jetson
@pytest.mark.slow
@pytest.mark.parametrize("concurrency_level", [1, 2, 4, 6, 8, 12, 16])
def test_concurrency_matrix(integration_config, registered_voice, concurrency_level):
    total_requests = concurrency_level * integration_config.repeats
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [
            executor.submit(run_single_request, integration_config, registered_voice["voice_id"], concurrency_level, index)
            for index in range(total_requests)
        ]
        for future in as_completed(futures):
            item = future.result()
            if item.get("ok"):
                results.append(item)
            else:
                failures.append(item)

    avg_total_seconds = None
    avg_first_chunk_seconds = None
    max_cer = None
    avg_cer = None
    if results:
        avg_total_seconds = sum(item["synth"]["total_seconds"] for item in results) / len(results)
        avg_first_chunk_seconds = sum(
            (item["synth"]["first_chunk_seconds"] or item["synth"]["total_seconds"]) for item in results
        ) / len(results)
        max_cer = max(item["comparison"]["cer"] for item in results)
        avg_cer = sum(item["comparison"]["cer"] for item in results) / len(results)

    summary = {
        "concurrency_level": concurrency_level,
        "total_requests": total_requests,
        "success_count": len(results),
        "failure_count": len(failures),
        "avg_total_seconds": avg_total_seconds,
        "avg_first_chunk_seconds": avg_first_chunk_seconds,
        "max_cer": max_cer,
        "avg_cer": avg_cer,
    }
    write_json(
        integration_config.output_dir / "concurrency" / f"concurrency_{concurrency_level}.json",
        {"summary": summary, "results": results, "failures": failures},
    )
