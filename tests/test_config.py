import pytest

from inference_server.config import ServerConfig


def test_valid_config() -> None:
    config = ServerConfig(model="model", served_model_name="local", max_num_seqs=32)
    config.validate()


def test_invalid_gpu_fraction() -> None:
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        ServerConfig(
            model="m",
            served_model_name="m",
            gpu_memory_utilization=1.1,
        ).validate()


def test_invalid_port() -> None:
    with pytest.raises(ValueError, match="port"):
        ServerConfig(model="m", served_model_name="m", port=70000).validate()


def test_invalid_boolean_environment_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_PREFIX_CACHING", "sometimes")
    with pytest.raises(ValueError, match="ENABLE_PREFIX_CACHING"):
        ServerConfig.from_env()


def test_false_boolean_environment_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_PREFIX_CACHING", "false")
    assert ServerConfig.from_env().enable_prefix_caching is False
