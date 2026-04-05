from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from nanovllm.utils.context import Context


def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:
    return cache.view(-1, cache.size(-2), cache.size(-1))


def _expand_kv_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    assert num_heads % x.size(1) == 0
    repeat = num_heads // x.size(1)
    return x.repeat_interleave(repeat, dim=1)


def _store_kvcache_impl(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor | None,
) -> None:
    if slot_mapping is None or slot_mapping.numel() == 0:
        return
    flat_k_cache = _flatten_cache(k_cache)
    flat_v_cache = _flatten_cache(v_cache)
    valid = slot_mapping >= 0
    if not torch.any(valid):
        return
    slots = slot_mapping[valid].to(dtype=torch.long)
    flat_k_cache.index_copy_(0, slots, key[valid])
    flat_v_cache.index_copy_(0, slots, value[valid])


def _gather_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    if seq_len == 0:
        return cache.new_empty((0, cache.size(-2), cache.size(-1)))
    block_size = cache.size(1)
    flat_cache = _flatten_cache(cache)
    token_positions = torch.arange(seq_len, device=cache.device, dtype=torch.long)
    block_indices = block_table.index_select(0, token_positions // block_size).to(dtype=torch.long)
    slot_indices = block_indices * block_size + (token_positions % block_size)
    return flat_cache.index_select(0, slot_indices)


def _causal_mask(
    q_len: int,
    k_len: int,
    prefix_len: int,
    device: torch.device,
) -> torch.Tensor:
    q_positions = prefix_len + torch.arange(q_len, device=device, dtype=torch.long)
    k_positions = torch.arange(k_len, device=device, dtype=torch.long)
    return k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)


def _attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    prefix_len: int,
    output_3d: bool,
) -> torch.Tensor:
    if q.numel() == 0:
        shape = (0, q.size(1), q.size(2)) if not output_3d else (0, 1, q.size(1), q.size(2))
        return q.new_empty(shape)
    q_heads = q.transpose(0, 1)
    k_heads = _expand_kv_heads(k, q.size(1)).transpose(0, 1)
    v_heads = _expand_kv_heads(v, q.size(1)).transpose(0, 1)
    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
    mask = _causal_mask(q.size(0), k.size(0), prefix_len, q.device)
    scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_heads).transpose(0, 1).contiguous()
    return output.unsqueeze(1) if output_3d else output


@dataclass(frozen=True)
class AttentionBackend:
    name: str

    def store_kvcache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        _store_kvcache_impl(key, value, k_cache, v_cache, slot_mapping)

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        raise NotImplementedError


class DefaultAttentionBackend(AttentionBackend):

    def __init__(self) -> None:
        super().__init__(name="default")

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func

        if context.block_tables is not None:
            k, v = k_cache, v_cache
        return flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=scale,
            causal=True,
            block_table=context.block_tables,
        )

    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_with_kvcache

        return flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=scale,
            causal=True,
        )


class SwEmulatorAttentionBackend(AttentionBackend):

    def __init__(self) -> None:
        super().__init__(name="sw_emulator")

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        if context.block_tables is None:
            return _attention_reference(q, k, v, scale, prefix_len=0, output_3d=False)

        outputs = []
        q_starts = context.cu_seqlens_q.tolist()
        k_starts = context.cu_seqlens_k.tolist()
        for idx in range(len(q_starts) - 1):
            q_start, q_end = q_starts[idx], q_starts[idx + 1]
            k_start, k_end = k_starts[idx], k_starts[idx + 1]
            q_seq = q[q_start:q_end]
            k_len = k_end - k_start
            prefix_len = k_len - q_seq.size(0)
            block_table = context.block_tables[idx]
            valid_blocks = block_table[block_table >= 0]
            k_seq = _gather_paged_cache(k_cache, valid_blocks, k_len)
            v_seq = _gather_paged_cache(v_cache, valid_blocks, k_len)
            outputs.append(_attention_reference(q_seq, k_seq, v_seq, scale, prefix_len, output_3d=False))
        return torch.cat(outputs, dim=0) if outputs else q.new_empty((0, num_heads, q.size(-1)))

    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
        scale: float,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        outputs = []
        context_lens = context.context_lens.tolist()
        for idx, seq_len in enumerate(context_lens):
            block_table = context.block_tables[idx]
            valid_blocks = block_table[block_table >= 0]
            k_seq = _gather_paged_cache(k_cache, valid_blocks, seq_len)
            v_seq = _gather_paged_cache(v_cache, valid_blocks, seq_len)
            outputs.append(_attention_reference(q[idx:idx + 1], k_seq, v_seq, scale, seq_len - 1, output_3d=True))
        return torch.cat(outputs, dim=0) if outputs else q.new_empty((0, 1, num_heads, q.size(-1)))


@lru_cache(maxsize=None)
def get_attention_backend(name: str) -> AttentionBackend:
    backends = {
        "default": DefaultAttentionBackend,
        "sw_emulator": SwEmulatorAttentionBackend,
    }
    if name not in backends:
        raise ValueError(f"Unsupported attention backend: {name}")
    return backends[name]()
