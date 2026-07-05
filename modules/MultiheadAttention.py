import torch
from torch import nn, Tensor
from torch.autograd.function import FunctionCtx
import math
from .Linear import Linear
from .RotaryPositionalEmbeddings import RotaryPositionEmbeddings

def _to_multihead(input: Tensor, emb_dim: int, num_heads: int) -> Tensor:
    input = input.unsqueeze(0) if len(input.shape) == 1 else input
    batch_dims = input.shape[:-2]
    seq = input.shape[-2]
    head_dim = emb_dim // num_heads
    multihead_dim = batch_dims + (seq, num_heads, head_dim)
    batch_idx = tuple(range(len(batch_dims)))
    return input.view(multihead_dim).permute(batch_idx + (-2, -3, -1))

def _from_multihead(input: Tensor, emb_dim: int) -> Tensor:
    batch_dims = input.shape[:-3]
    seq = input.shape[-2]
    multihead_dim = batch_dims + (seq, emb_dim)
    batch_idx = tuple(range(len(batch_dims)))
    return input.permute(batch_idx + (-2, -3, -1)).reshape(multihead_dim)


# CORE CHANGE: Function now takes Q, K, V already in multihead format (B, heads, seq, head_dim).
# Projections and _to_multihead are moved to the Module. W_q/W_k/W_v/W_o removed from signature.
class MultiheadAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        Q: Tensor,                          # (B, num_heads, seq_q, head_dim) — already projected
        K: Tensor,                          # (B, num_heads, seq_k, head_dim)
        V: Tensor,                          # (B, num_heads, seq_k, head_dim)
        emb_dim: int,
        num_heads: int,
        key_padding_mask: Tensor | None=None,
        attn_mask: Tensor | None=None,
        is_causal: bool=False
    ):
        ctx.num_heads = num_heads
        ctx.emb_dim = emb_dim
        ctx.batch_dims = Q.shape[:-3]       # CORE CHANGE: was query.shape[:-2]
        N = math.prod(ctx.batch_dims)

        seq_q = Q.shape[-2]
        seq_k = K.shape[-2]

        A = Q @ K.mT

        if key_padding_mask is not None:
            if key_padding_mask.shape != (ctx.batch_dims + (seq_k,)):
                raise KeyError(f"key_padding_mask must have shape batch_dims + (seq_k,), but got {key_padding_mask.shape}")
            elif key_padding_mask.dtype == torch.bool:
                A = A.masked_fill(key_padding_mask.view(*ctx.batch_dims, 1, 1, seq_k), float('-inf'))
            else:
                A += key_padding_mask.view(*ctx.batch_dims, 1, 1, seq_k)

        if attn_mask is not None:
            if attn_mask.shape == (seq_q, seq_k):
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.shape == (N * num_heads, seq_q, seq_k):
                attn_mask = attn_mask.reshape(ctx.batch_dims + (num_heads, seq_q, seq_k))
            else:
                raise KeyError(f"attn_mask must have shape (seq_q, seq_k) or (N * num_heads, seq_q, seq_k), but got {attn_mask.shape}")
            A = A + attn_mask if attn_mask.dtype != torch.bool else A.masked_fill(attn_mask, float('-inf'))

        if is_causal:
            mask = (float('-inf') * torch.ones((seq_q, seq_k))).triu(1)
            A += mask

        attn_map = torch.softmax(A / math.sqrt(K.shape[-1]), dim=-1)
        attn_output = _from_multihead(attn_map @ V, emb_dim)

        # CORE CHANGE: save Q, K, V directly — no recomputation needed in backward
        ctx.save_for_backward(Q, K, V, attn_map)

        return attn_output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_upstream: Tensor):
        # CORE CHANGE: backward is significantly simpler — no W_q/W_k/W_v/W_o gradients,
        # no recomputation of Q/K/V. Just returns grad_Q, grad_K, grad_V in multihead format;
        # autograd propagates the rest back through _to_multihead and the projections.
        Q, K, V, attn_map = ctx.saved_tensors

        grad_attn_output = _to_multihead(grad_upstream, ctx.emb_dim, ctx.num_heads)

        grad_V = attn_map.mT @ grad_attn_output if ctx.needs_input_grad[2] else None

        grad_attn_map = grad_attn_output @ V.mT
        grad_A = attn_map * (grad_attn_map - (grad_attn_map * attn_map).sum(dim=-1, keepdim=True)) / math.sqrt(K.shape[-1])

        grad_Q = grad_A @ K   if ctx.needs_input_grad[0] else None
        grad_K = grad_A.mT @ Q if ctx.needs_input_grad[1] else None

        return (
            grad_Q,   # 0: Q
            grad_K,   # 1: K
            grad_V,   # 2: V
            None,     # 3: emb_dim
            None,     # 4: num_heads
            None,     # 5: key_padding_mask
            None,     # 6: attn_mask
            None,     # 7: is_causal
        )


class MultiheadAttention(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            rope_seq_len: int | None=None,
            bias: bool=True,
            add_bias_kv: bool=False,
            device: torch.device=None,
            kdim: int=None,
            vdim: int=None,
            dtype: torch.dtype=None
        ):
        super().__init__()

        assert embed_dim % num_heads == 0, "num_heads must divide embed_dim"
        self.num_heads = num_heads
        self.emb_dim = embed_dim
        self.proj_q = Linear(embed_dim, embed_dim,              bias=bias,       device=device, dtype=dtype)
        self.proj_k = Linear(embed_dim, kdim or embed_dim,     bias=add_bias_kv, device=device, dtype=dtype)
        self.proj_v = Linear(embed_dim, vdim or embed_dim,     bias=add_bias_kv, device=device, dtype=dtype)
        self.proj_o = Linear(embed_dim, embed_dim,              bias=bias,       device=device, dtype=dtype)
        
        self.rope = None
        if rope_seq_len:
            self.rope = RotaryPositionEmbeddings(dim=embed_dim // num_heads, max_seq_len=rope_seq_len)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Tensor | None=None,
            attn_mask: Tensor | None=None,
            is_causal: bool=False
    ):
        Q = _to_multihead(self.proj_q(query), self.emb_dim, self.num_heads)
        K = _to_multihead(self.proj_k(key),   self.emb_dim, self.num_heads)
        V = _to_multihead(self.proj_v(value), self.emb_dim, self.num_heads)

        if self.rope is not None:
            Q, K = self.rope(Q, K)

        attn_output = MultiheadAttentionFunction.apply(
            Q, K, V,
            self.emb_dim, self.num_heads,
            key_padding_mask, attn_mask, is_causal,
        )

        return self.proj_o(attn_output)
