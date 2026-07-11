import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from typing import Callable, Tuple
import argparse

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

    parser = argparse.ArgumentParser(description="Benchmark naive attention vs flash attention")
    parser.add_argument("-b", "--batch", type=int, default=2, help="tensor batch size")
    parser.add_argument("-h", "--heads", type=int, default=8, help="tensor head size")
    parser.add_argument("-s", "--seq_len", type=int, default=2048, help="tensor sequence length")
    parser.add_argument("-d", "--dims", type=int, default=32, help="tensor feature dim size")

    args = parser.parse_args()

    batch_size = args.batch
    n_heads = args.heads
    N = args.seq_len
    DIM = args.dims
    
    mha_naive = lambda Q, K, V: MultiheadAttentionFunction.apply(
            Q, K, V,
            DIM, n_heads, None, None, False, False)
    
    mha_flash = lambda Q, K, V: MultiheadAttentionFunction.apply(
            Q, K, V,
            DIM, n_heads, None, None, False, True)

    naive_func = run_4d(3, (batch_size, n_heads, N, DIM), mha_naive, torch.device("cuda"))
    flash_func = run_4d(3, (batch_size, n_heads, N, DIM), mha_flash, torch.device("cuda"))
    
    with torch.cuda.nvtx.range("naive"):
        avg_time_naive = benchmark(naive_func)
    
    with torch.cuda.nvtx.range("flash"):
        avg_time_flash = benchmark(flash_func)

    speedup = avg_time_naive / avg_time_flash
    print(f"Config: batch={batch_size}, heads={n_heads}, seq_len={N}, head_dim={DIM}")
    print(f"  Naive attention : {avg_time_naive:.4f} ms")
    print(f"  Flash attention : {avg_time_flash:.4f} ms")
    print(f"  Speedup         : {speedup:.2f}x")