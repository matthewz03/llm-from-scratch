import torch
from torch import nn, Tensor
from torch.autograd.function import FunctionCtx
import math

class LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, input_data: Tensor, weight: Tensor, bias: Tensor):
        ctx.save_for_backward(input_data, weight, bias)

        out = input_data @ weight.T
        if bias is not None:
            out += bias
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_upstream: Tensor):
        input_data, weight, bias = ctx.saved_tensors

        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_upstream @ weight # (B x out) @ (out x in) -> B x in
        if ctx.needs_input_grad[1]:
            grad_weight = grad_upstream.transpose(-2, -1) @ input_data # (out x B) @ (B x in) -> out x in
            
            num_dims = len(grad_upstream.size()) - len(weight.size())
            if num_dims:
                sum_dims = tuple(range(num_dims))
                grad_weight = grad_weight.sum(dim=sum_dims)
        if bias is not None and ctx.needs_input_grad[2]:
            num_dims = len(grad_upstream.size()) - len(bias.size())

            # (B x out) -> out
            grad_bias = grad_upstream.sum(dim=tuple(range(num_dims))) if num_dims else grad_upstream.clone()

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool=True, 
        device: torch.device=None,
        dtype=None
    ):
        super().__init__()

        self.weight = nn.Parameter(data=torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight.data)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(data=torch.empty(out_features, device=device, dtype=dtype))
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias.data, -bound, bound)

    def forward(self, input: Tensor):
        return LinearFunction.apply(input, self.weight, self.bias)

