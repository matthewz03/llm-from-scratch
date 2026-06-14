from torch import nn, Tensor

from .LayerNorm import LayerNorm
from .Embedding import Embedding
from .TransformerBlock import TransformerBlock
from .Linear import Linear

class LanguageModel(nn.Module):

    def __init__(
            self, 
            d_model: int, 
            vocab_size: int, 
            n_blocks: int, 
            dim_feedforward: int, 
            n_attn_heads: int, 
            norm_bias: bool=False,
            attn_bias: bool=False,
            ffn_bias: bool=False,
            lm_head_bias: bool=False,
            rope_seq_len: int | None=4096):
        super().__init__()

        self.emb = Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, 
                n_heads=n_attn_heads, 
                rope_seq_len=rope_seq_len, 
                dim_feedforward=dim_feedforward, 
                norm_bias=norm_bias, 
                attn_bias=attn_bias, 
                ffn_bias=ffn_bias
            )
            for _ in range(n_blocks)
        ])

        self.lm_norm = LayerNorm(normalized_shape=d_model, bias=norm_bias)
        self.lm_head = Linear(d_model, vocab_size, bias=lm_head_bias)

    def forward(self, input: Tensor):
        embeddings = self.emb(input)

        for block in self.blocks:
            embeddings = block(embeddings)
        
        logits = self.lm_head(self.lm_norm(embeddings))
        # for decode, expect embeddings to be of shape (..., N, D)
        # we need the last token i.e. (..., -1, D)
        # final_token = embeddings[..., -1, :]
        # logits = self.lm_head(final_token)

        return logits