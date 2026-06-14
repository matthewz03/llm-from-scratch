import torch
from torch import nn, Tensor
from torch.autograd.function import FunctionCtx

class CrossEntropyLossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, logits: Tensor, target: Tensor, reduction: str='mean'):
        # logits are logits: C or B, C or B, C, ...
        # target: (), or B or B, ...
        # for now just the first 2 cases
        assert logits.shape == target.shape or logits.shape[:-1] == target.shape

        ctx.target_is_index = logits.shape != target.shape
        ctx.reduction = reduction
        ctx.numel = logits.shape[:-1].numel()

        log_probs = logits.log_softmax(dim=-1)
        probs = log_probs.exp()


        class_probs = log_probs.gather(-1, target.unsqueeze(-1)) if ctx.target_is_index else (log_probs * target).sum(dim=-1)
        loss = -class_probs.sum() if reduction == 'sum' else -class_probs.mean()

        ctx.save_for_backward(probs, target)

        return loss
    
    @staticmethod
    def backward(ctx, grad_upstream: Tensor):
        '''
        if target is indices:
            only the indexed parts of the logits impact loss
            
        '''
        probs, target = ctx.saved_tensors

        if ctx.reduction == 'mean':
            grad_upstream = grad_upstream / ctx.numel

        class_targets = (
            torch.zeros_like(probs).scatter_(-1, target.unsqueeze(-1), 1) 
            if ctx.target_is_index else 
            target 
        )
        grad_logits = grad_upstream * (probs - class_targets)

        return grad_logits, None, None


class CrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str='mean'):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError(f"reduction must be 'mean' or 'sum', but got {reduction}")
        self.reduction = reduction

    def forward(self, logitss: Tensor, target: Tensor):
        return CrossEntropyLossFunction.apply(logitss, target, self.reduction)