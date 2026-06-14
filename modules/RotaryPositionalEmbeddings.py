import torch
from torch import nn, Tensor

class RotaryPositionEmbeddings(nn.Module):

    def __init__(self, dim: int, max_seq_len: int, base: int=10000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for RoPE"

        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        theta_i = self.base ** (-2 * torch.arange(dim // 2) / self.dim)
        theta_i = theta_i.repeat_interleave(2)
        
        m = torch.arange(max_seq_len)
        m_theta = m.outer(theta_i)

        # cos, sin: (1, N, D)
        self.register_buffer(
            'cos',
            m_theta.cos().unsqueeze(0)
        )
        sin = m_theta.sin().unsqueeze(0)
        sin[..., ::2] *= -1
        self.register_buffer(
            'sin',
            sin
        )


    def forward(self, query: Tensor, key: Tensor):

        q_seq_len, k_seq_len = query.shape[-2], key.shape[-2]

        assert q_seq_len <= self.max_seq_len, f"Sequence length {q_seq_len} exceeds maximum {self.max_seq_len}"
        
        # (B, N, D) * (1, N, D)
        query = (
            query * self.cos[:, :q_seq_len, :] + 
            query.view(query.shape[:-1] + (-1, 2)).flip(-1).flatten(start_dim=-2) 
            * self.sin[:, :q_seq_len, :]
        )
        key = (
            key * self.cos[:, :k_seq_len, :] + 
            key.view(key.shape[:-1] + (-1, 2)).flip(-1).flatten(start_dim=-2) 
            * self.sin[:, :k_seq_len, :]
        )

        return query, key