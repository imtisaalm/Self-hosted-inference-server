from inference_server.metrics import scheduler_metrics


def test_scheduler_metrics() -> None:
    text = """
vllm:num_requests_running{model_name="m"} 4
vllm:num_requests_waiting{model_name="m"} 2
vllm:kv_cache_usage_perc 0.375
"""
    result = scheduler_metrics(text)
    assert result.running == 4
    assert result.waiting == 2
    assert result.kv_cache_usage == 0.375
