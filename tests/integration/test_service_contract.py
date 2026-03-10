import pytest
import requests

from tests.integration.conftest import write_json


@pytest.mark.integration
@pytest.mark.jetson
def test_models_endpoint(integration_config):
    response = requests.get(f"{integration_config.tts_base_url}/v1/models", timeout=30)
    response.raise_for_status()
    payload = response.json()
    write_json(
        integration_config.output_dir / "smoke" / "models_endpoint.json",
        {"endpoint": "/v1/models", "status_code": response.status_code, "payload": payload},
    )
    assert payload["data"][0]["id"] == integration_config.model_id


@pytest.mark.integration
@pytest.mark.jetson
def test_metrics_endpoint(integration_config):
    response = requests.get(f"{integration_config.tts_base_url}/metrics", timeout=30)
    response.raise_for_status()
    payload = response.json()
    write_json(
        integration_config.output_dir / "smoke" / "metrics_endpoint.json",
        {"endpoint": "/metrics", "status_code": response.status_code, "payload": payload},
    )
    assert "queue" in payload
