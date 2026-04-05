import os
import socket
from dataclasses import dataclass
from transformers import AutoConfig


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


@dataclass
class Config:
    model: str
    backend: str = "default"
    dist_init_method: str | None = None
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.backend = self.backend.lower()
        assert self.backend in {"default", "sw_emulator"}
        self.hf_config = AutoConfig.from_pretrained(self.model)
        setattr(self.hf_config, "backend", self.backend)
        if self.dist_init_method is None:
            self.dist_init_method = f"tcp://127.0.0.1:{_find_free_port()}"
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
