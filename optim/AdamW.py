import torch
from collections.abc import Iterable
from typing import Tuple

class AdamW(torch.optim.Optimizer):
    '''
    AdamW optimizer implementation in PyTorch. 
    AdamW is a variant of the Adam optimizer that decouples 
    weight decay from the gradient update, allowing for 
    better regularization.
    
        The update rule for AdamW is as follows:
    m(t) = beta1 * m(t-1) + (1 - beta1) * grad(t)
    v(t) = beta2 * v(t-1) + (1 - beta2) * (grad(t) ** 2)

    m_hat = m(t) / (1 - ( beta1 ** (t) ))
    v_hat = v(t) / (1 - ( beta2 ** (t) ) )

    W = W - lr * m_hat / ( eps + sqrt(v_hat) )
    '''

    def __init__(self, params: Iterable, lr: float=1e-3, betas: Tuple[float]=(0.9, 0.999), eps: float=1e-8, weight_decay: float=0.0):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate, learning rate must be greater than 0 but got {lr}")
        if any(b < 0 for b in betas) or any(b > 1 for b in betas):
            raise ValueError(f"Invalid beta1, beta1 must be between 0 and 1 but got {betas}")
        defaults={
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }

        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['m'] = 0
                    state['v'] = 0
                    state['betas'] = (1, 1)
                
                state['m'] = beta1 * state['m'] + (1 - beta1) * p.grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * p.grad * p.grad
                state['betas'] = (state['betas'][0] * beta1, state['betas'][1] * beta2)

                m_update = state['m'] / (1 - state['betas'][0])
                v_update = state['v'] / (1 - state['betas'][1])

                with torch.no_grad():
                    p.data.add_(
                        m_update / (v_update.sqrt() + eps) + 
                        weight_decay * p, 
                        alpha=-lr
                    )
                