import torch
from torch import nn, Tensor
from torch.autograd.function import FunctionCtx

class ReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, inplace: bool):
        ctx.save_for_backward(input > 0)
        if inplace:
            input.clamp_(min=0)
            return input

        return torch.clamp(input, min=0)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_upstream: Tensor):
        mask, = ctx.saved_tensors
        return torch.where(mask, grad_upstream, 0), None


class ReLU(nn.Module):

    def __init__(self, inplace: bool=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor):
        return ReLUFunction.apply(input, self.inplace)