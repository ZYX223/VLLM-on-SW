import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nanovllm.backends import get_attention_backend
from nanovllm.utils.context import Context


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, prefix_len: int, output_3d: bool):
    num_heads = q.size(1)
    if k.size(1) != num_heads:
        repeat = num_heads // k.size(1)
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    q_heads = q.transpose(0, 1)
    k_heads = k.transpose(0, 1)
    v_heads = v.transpose(0, 1)
    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
    q_positions = prefix_len + torch.arange(q.size(0), dtype=torch.long)
    k_positions = torch.arange(k.size(0), dtype=torch.long)
    mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_heads).transpose(0, 1).contiguous()
    return output.unsqueeze(1) if output_3d else output


class SwEmulatorAttentionBackendTest(unittest.TestCase):

    def setUp(self):
        self.backend = get_attention_backend("sw_emulator")

    def test_prefill_without_prefix_cache_matches_reference(self):
        q = torch.tensor(
            [
                [[1.0, 0.0], [0.5, -0.5]],
                [[0.0, 1.0], [1.0, 0.5]],
                [[1.0, 1.0], [-0.5, 0.5]],
            ]
        )
        k = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
            ]
        )
        v = torch.tensor(
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
                [[5.0, 6.0]],
            ]
        )
        context = Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
            max_seqlen_q=3,
            max_seqlen_k=3,
        )

        output = self.backend.prefill(q, k, v, torch.empty(0), torch.empty(0), context, scale=1.0, num_heads=2, num_kv_heads=1)
        expected = reference_attention(q, k, v, scale=1.0, prefix_len=0, output_3d=False)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_prefill_with_prefix_cache_reads_paged_cache(self):
        k_cache = torch.zeros(2, 2, 1, 2)
        v_cache = torch.zeros(2, 2, 1, 2)

        prefix_k = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        prefix_v = torch.tensor([[[10.0, 1.0]], [[20.0, 2.0]]])
        self.backend.store_kvcache(
            prefix_k,
            prefix_v,
            k_cache,
            v_cache,
            torch.tensor([0, 1], dtype=torch.int32),
        )

        q = torch.tensor([[[1.0, 0.0], [0.5, 0.5]], [[0.0, 1.0], [1.0, -1.0]]])
        new_k = torch.tensor([[[1.0, 1.0]], [[-1.0, 1.0]]])
        new_v = torch.tensor([[[30.0, 3.0]], [[40.0, 4.0]]])
        context = Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 4], dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_k=4,
            slot_mapping=torch.tensor([2, 3], dtype=torch.int32),
            block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        )

        self.backend.store_kvcache(new_k, new_v, k_cache, v_cache, context.slot_mapping)
        output = self.backend.prefill(q, new_k, new_v, k_cache, v_cache, context, scale=0.5, num_heads=2, num_kv_heads=1)
        full_k = torch.cat([prefix_k, new_k], dim=0)
        full_v = torch.cat([prefix_v, new_v], dim=0)
        expected = reference_attention(q, full_k, full_v, scale=0.5, prefix_len=2, output_3d=False)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_decode_uses_full_context_from_block_table(self):
        k_cache = torch.zeros(2, 2, 1, 2)
        v_cache = torch.zeros(2, 2, 1, 2)
        cached_k = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]])
        cached_v = torch.tensor([[[5.0, 1.0]], [[6.0, 2.0]], [[7.0, 3.0]]])
        self.backend.store_kvcache(
            cached_k,
            cached_v,
            k_cache,
            v_cache,
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )

        q = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
        context = Context(
            is_prefill=False,
            context_lens=torch.tensor([3], dtype=torch.int32),
            block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        )

        output = self.backend.decode(q, k_cache, v_cache, context, scale=1.0, num_heads=2, num_kv_heads=1)
        expected = reference_attention(q, cached_k, cached_v, scale=1.0, prefix_len=2, output_3d=True)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
