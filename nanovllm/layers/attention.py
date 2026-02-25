import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

import sys
from pathlib import Path

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
    print(f"zml: 1. key.shape={key.shape}, value.shape={value.shape}, k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}, {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
    print(f"zml: 2. key.shape={key.shape}, value.shape={value.shape}, k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}, {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")



class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        print(f"zml: in init, k_cache.shape={self.k_cache.shape}, v_cache.shape={self.v_cache.shape}")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        print(f"zml: start, q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}, {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        print(f"zml: 1. k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}, {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        print(f"zml: 2. k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}, {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        print(f"zml: 2. q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}, {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                print(f"zml: hit prefix_cache")
                k, v = k_cache, v_cache
                print(f"zml: hit prefix_cache 1. k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}, {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
            print(f"zml: 3. q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}, {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")

            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            print(f"zml: Attention.decode q.shape={q.shape}, k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}, context.context_lens={context.context_lens}, {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
            print(f"zml: Attention.decode, q.unsqueeze(1)={q.unsqueeze(1).shape}")
            print(f"zml: Attention.decode, q.shape={q.shape}")
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
