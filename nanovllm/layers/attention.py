import torch
from torch import nn

from nanovllm.backends import get_attention_backend
from nanovllm.utils.context import get_context


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        backend_name: str = "default",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.backend = get_attention_backend(backend_name)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.backend.store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            return self.backend.prefill(q, k, v, k_cache, v_cache, context, self.scale, self.num_heads, self.num_kv_heads)
        return self.backend.decode(q, k_cache, v_cache, context, self.scale, self.num_heads, self.num_kv_heads)
