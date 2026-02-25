import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
import sys
from pathlib import Path

class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
# 在 Qwen3Attention.__init__ 中，get_rope 调用前添加
        if True:
            print(f"head_dim: {self.head_dim}, type: {type(self.head_dim)}")
            print(f"rope_theta: {rope_theta}, type: {type(rope_theta)}")
            print(f"max_position_embeddings: {max_position}, type: {type(max_position)}")
            print(f"rope_scaling: {rope_scaling}, type: {type(rope_scaling)}")
        # 检查是否有其他可能的字典参数
        #print(f"config attributes: {[k for k, v in config.__dict__.items() if isinstance(v, dict)]}")
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        print(f"zml: run into finish {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})(Qwen3Attention)")
        qkv = self.qkv_proj(hidden_states)
        print(f"zml: qkv.shape={qkv.shape}")
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        print(f"zml: q_split.shape={q.shape}, k_split.shape={k.shape}, v_split.shape={v.shape}")
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        print(f"zml: q_view.shape={q.shape}, k_view.shape={k.shape}, v_view.shape={v.shape}")
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)  # 获取rope的参数
        print(f"zml: q_rota={q.shape}, k_rota={k.shape}")
        o = self.attn(q, k, v)
        print(f"zml: o.shape={o.shape}")
        output = self.o_proj(o.flatten(1, -1))
        print(f"zml: output.shape={output.shape}")
        print(f"zml: run into finish {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}")
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()
        print(f"zml: run into finish {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}")

    def forward(self, x):
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})(Qwen3MLP)")
        print(f"zml: x.shape={x.shape}")
        gate_up = self.gate_up_proj(x)
        print(f"zml: gate_up.shape={gate_up.shape}")
        x = self.act_fn(gate_up)
        print(f"zml: after act_fn, x.shape={x.shape}")
        x = self.down_proj(x)
        print(f"zml: after down_proj, x.shape={x.shape}")
        print(f"zml: run into finish {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}(Qwen3MLP)")
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})(Qwen3DecoderLayer)")
        super().__init__()
        ## zml add for debug
        rope_scaling=getattr(config, "rope_scaling", None)
        #print(f"zml: rope_scaling={rope_scaling},type(rope_scaling)={type(rope_scaling)}")
        #print(f"zml: config={config}")
        
        
        
        ################################
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            #rope_scaling=getattr(config, "rope_scaling", None),
            rope_scaling=None,
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        print(f"zml: run into finish {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})(Qwen3DecoderLayer)")
        if residual is None:
            print(f"zml: in Qwen3DecoderLayer.forward, before input_layernorm, hidden_states.shape={hidden_states.shape}, residual=None")
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        print(f"zml: in Qwen3DecoderLayer.forward, after input_layernorm, hidden_states.shape={hidden_states.shape}, residual.shape={residual.shape}, positions.shape={positions.shape}")
        hidden_states = self.self_attn(positions, hidden_states)
        print(f"zml: in Qwen3DecoderLayer.forward, after self_attn, hidden_states.shape={hidden_states.shape}, positions.shape={positions.shape}")
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        print(f"zml: in Qwen3DecoderLayer.forward, after post_attention_layernorm, hidden_states.shape={hidden_states.shape}, residual.shape={residual.shape}")
        hidden_states = self.mlp(hidden_states)
        print(f"zml: in Qwen3DecoderLayer.forward, after mlp, hidden_states.shape={hidden_states.shape}")
        print(f"zml: run into finish {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}")
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        print(f"zml: run into finish {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        hidden_states = self.embed_tokens(input_ids)
        print(f"zml: positions.shape={positions.shape}, input_ids.shape={input_ids.shape}, hidden_states.shape={hidden_states.shape}")
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        print(f"zml: run into finish {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}")
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}:{sys._getframe().f_lineno}({sys._getframe().f_code.co_name})")
        return self.lm_head(hidden_states)
