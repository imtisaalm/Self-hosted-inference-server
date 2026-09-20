import pytest

from inference_server.config import ServerConfig


def test_valid_config() -> None:
    config = ServerConfig(model="model", served_model_name="local", max_num_seqs=32)
    config.validate()


def test_invalid_gpu_fraction() -> None:
    with pytest.raises(ValueError):
        ServerConfig(model="m", served_model_name="m", gpu_memory_utilization=1.1).validate()
