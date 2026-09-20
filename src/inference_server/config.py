from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ServerConfig:
    model: str
    served_model_name: str
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    max_num_seqs: int = 64
    max_num_batched_tokens: int | None = None
    tensor_parallel_size: int = 1
    kv_cache_dtype: str = "auto"
    enable_prefix_caching: bool = True

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_model_len <= 0 or self.max_num_seqs <= 0:
            raise ValueError("max_model_len and max_num_seqs must be positive")
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")

    @classmethod
    def from_env(cls) -> "ServerConfig":
        def flag(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        batched = os.getenv("MAX_NUM_BATCHED_TOKENS")
        config = cls(
            model=os.getenv("MODEL", "Qwen/Qwen3-0.6B"),
            served_model_name=os.getenv("SERVED_MODEL_NAME", "local-model"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),
            max_model_len=int(os.getenv("MAX_MODEL_LEN", "8192")),
            max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "64")),
            max_num_batched_tokens=int(batched) if batched else None,
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
            kv_cache_dtype=os.getenv("KV_CACHE_DTYPE", "auto"),
            enable_prefix_caching=flag("ENABLE_PREFIX_CACHING", True),
        )
        config.validate()
        return config
