from torch import nn, Tensor
from .MultiheadAttention import MultiheadAttention
from .Linear import Linear
from .LayerNorm import LayerNorm
from .ReLU import ReLU


class TransformerBlock(nn.Module):

    def __init__(
            self, 
            d_model: int, 
            n_heads: int, 
            rope_seq_len: int | None=4096, 
            dim_feedforward: int=2048, 
            norm_bias: bool=False,
            attn_bias: bool=False,
            ffn_bias: bool=False,
        ):
        super().__init__()

        self.norm1 = LayerNorm(normalized_shape=d_model, bias=norm_bias)
        self.mha = MultiheadAttention(embed_dim=d_model, num_heads=n_heads, rope_seq_len=rope_seq_len, bias=attn_bias)

        self.norm2 = LayerNorm(normalized_shape=d_model, bias=norm_bias)
        self.ffn1 = Linear(d_model, dim_feedforward, bias=ffn_bias)
        self.relu = ReLU()
        self.ffn2 = Linear(dim_feedforward, d_model, bias=ffn_bias)

    def forward(self, input: Tensor, is_causal: bool=True):

        norm_input1 = self.norm1(input)
        residual_mha = self.mha(norm_input1, norm_input1, norm_input1, is_causal=is_causal)
        input = input + residual_mha

        norm_input2 = self.norm2(input)
        residual_ffn = self.ffn2(
            self.relu(
                self.ffn1(norm_input2)
            )
        )
        return input + residual_ffn