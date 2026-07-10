import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from typing import Callable, Tuple
from modules.MultiheadAttention import MultiheadAttentionFunction


def benchmark(func: Callable, n_trials: int=10, n_warmups: int=5):

    for _ in range(n_warmups):
        func()

    torch.cuda.synchronize()

    times = []
    for i in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        func()
        end.record()

        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    avg_times = sum(times) / len(times)
    return avg_times

def run_4d(n_args: int, size: Tuple, func: Callable, device: torch.device | None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = [torch.randn(size, device=device) for _ in range(n_args)]
    return lambda : func(*args)

if __name__ == "__main__":

    batch_size = 32
    n_heads = 64
    N = 128
    DIM = 64
    mha_naive = lambda Q, K, V: MultiheadAttentionFunction.apply(
            Q, K, V,
            DIM, n_heads, flash_attn=False)
    
    mha_flash = lambda Q, K, V: MultiheadAttentionFunction.apply(
            Q, K, V,
            DIM, n_heads, flash_attn=True)

    naive_func = run_4d(3, (batch_size, n_heads, N, DIM), mha_naive, torch.device("cuda"))
    flash_func = run_4d(3, (batch_size, n_heads, N, DIM), mha_flash, torch.device("cuda"))
    
    avg_time_naive = benchmark(naive_func)
    avg_time_flash = benchmark(flash_func)

    print(f"Average time for MultiheadAttentionFunction.apply with batch_size={batch_size}, n_heads={n_heads}, N={N}, DIM={DIM}: {avg_time_naive:.4f} ms")
    print(f"Average time for MultiheadAttentionFunction.apply with batch_size={batch_size}, n_heads={n_heads}, N={N}, DIM={DIM}: {avg_time_flash:.4f} ms")