import torch
from torch import nn, Tensor
from torch.autograd.function import FunctionCtx

from typing import List

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx, 
        input: Tensor, 
        gamma: Tensor,
        beta: Tensor,
        normalized_shape: int | List[int] | torch.Size, 
        eps: float
    ):
        normalized_shape = list([normalized_shape] if isinstance(normalized_shape, int) else normalized_shape)
        num_dims = len(normalized_shape)
        assert torch.Size(normalized_shape) == input.shape[-num_dims:], "normalized_shape must match trailing dimensions of input"
        ctx.normalized_shape = normalized_shape
        ctx.num_dims = num_dims
        ctx.eps = eps

        ctx.dims = tuple(range(-num_dims, 0))
        
        mean = input.mean(dim=ctx.dims, keepdim=True)
        var = input.var(dim=ctx.dims, correction=0, keepdim=True)

        delta = input - mean

        sigma = torch.sqrt(var + eps)
        z = delta / sigma
        y = z * gamma 
        if beta is not None:
            y += beta

        ctx.save_for_backward(gamma, beta, z, delta, sigma)

        return y
        
    @staticmethod
    def backward(ctx: FunctionCtx, grad_upstream: Tensor):
        gamma, beta, z, delta, sigma = ctx.saved_tensors
        
        grad_input, grad_gamma, grad_beta = None, None, None
        collapse_dims = tuple(range(len(z.shape) - len(ctx.normalized_shape)))
        H = grad_upstream.shape[-ctx.num_dims:].numel()

        if ctx.needs_input_grad[0]:
            
            grad_z = grad_upstream * gamma
            grad_x = grad_z / sigma
            grad_mean = - torch.sum(grad_z / sigma, dim=ctx.dims, keepdim=True) / H
            grad_v = - torch.sum( grad_z * delta / sigma.pow(3), dim=ctx.dims, keepdim=True)
            grad_var = grad_v * delta / H
            grad_input = grad_x + grad_mean + grad_var

        if ctx.needs_input_grad[1]:
            grad_gamma = grad_upstream * z
            if ctx.num_dims:
                grad_gamma = grad_gamma.sum(dim=collapse_dims)
        if beta is not None and ctx.needs_input_grad[2]:
            grad_beta = grad_upstream
            if ctx.num_dims:
                grad_beta = grad_upstream.sum(dim=collapse_dims)

        return grad_input, grad_gamma, grad_beta, None, None

class LayerNorm(nn.Module):

    def __init__(
            self, 
            normalized_shape: int | List[int] | torch.Size,
            eps: float=1e-5,
            dtype: torch.dtype=None,
            bias: bool=True
        ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
        self.beta = nn.Parameter(torch.zeros(normalized_shape, dtype=dtype)) if bias else None

    def forward(self, input: Tensor):
        return LayerNormFunction.apply(
            input, 
            self.gamma, 
            self.beta, 
            self.normalized_shape,
            self.eps
        )