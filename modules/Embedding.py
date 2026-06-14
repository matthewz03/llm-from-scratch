import torch
from torch import nn, Tensor
from torch.autograd.function import FunctionCtx

class EmbeddingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, embeddings: Tensor, indices: Tensor):
        # indices: (B, N)
        # embeddings: (V, D)
        # output: (B, N, D)
        ctx.save_for_backward(indices)
        ctx.emb_shape = embeddings.shape
        ctx.emb_dtype = embeddings.dtype
        ctx.emb_device = embeddings.device
        return embeddings[indices]

    @staticmethod
    def backward(ctx: FunctionCtx, grad_upstream: Tensor):
        # grad_upstream: (B, N, D)
        # embeddings: (V, D)
        indices, = ctx.saved_tensors
        
        grad_embeddings = None
        if ctx.needs_input_grad[0]:
            grad_embeddings = torch.zeros(size=ctx.emb_shape, dtype=ctx.emb_dtype, device=ctx.emb_device).index_add_(
                dim=0, 
                index=indices.flatten(),
                source=grad_upstream.flatten(end_dim=-2)    
            )
        return grad_embeddings, None


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype=None, device: torch.device=None):
        super().__init__()

        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))

        # initialize embeddings
        nn.init.normal_(self.embeddings, mean=0, std=1)

    def forward(self, input: Tensor):
        return EmbeddingFunction.apply(self.embeddings, input)