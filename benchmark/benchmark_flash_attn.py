import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import torch
from typing import Callable, Tuple
import argparse
import numpy as np
import matplotlib.pyplot as plt

from modules.MultiheadAttention import MultiheadAttentionFunction


def benchmark(func: Callable, n_trials: int=10, n_warmups: int=5, nvtx_label: str | None=None):

    for _ in range(n_warmups):
        func()

    torch.cuda.synchronize()

    times = []
    for i in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with torch.cuda.nvtx.range(nvtx_label or ""):
            start.record()
            func()
            end.record()

        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    avg_times = sum(times) / len(times)
    return avg_times

def plot_results(results: pd.DataFrame, fig_path: str | None):
    seq_lens   = results['seq_len'].tolist()
    naive_times = results['naive_time'].tolist()
    flash_times = results['flash_time'].tolist()

    x     = np.arange(len(seq_lens))
    width = 0.35
    pad   = max(naive_times + flash_times) * 0.015

    fig, ax = plt.subplots(figsize=(8, 5))
    for obj in (fig, ax):
        obj.set_facecolor('#fcfcfb')

    bars_naive = ax.bar(x - width / 2, naive_times, width, label='Naive', color='#2a78d6', linewidth=0, zorder=3)
    bars_flash = ax.bar(x + width / 2, flash_times, width, label='Flash', color='#1baf7a', linewidth=0, zorder=3)

    # direct labels required: aqua (#1baf7a) is sub-3:1 on white surface
    for bar in (*bars_naive, *bars_flash):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=8, color='#52514e')

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.set_xlabel('Sequence length', color='#52514e', labelpad=8)
    ax.set_ylabel('Time (ms)',        color='#52514e', labelpad=8)
    ax.set_title('Flash vs Naive Attention', color='#0b0b0b', fontsize=13, fontweight='bold', pad=12)

    ax.legend(frameon=False, labelcolor='#52514e')
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors='#898781')

    fig.tight_layout()
    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='#fcfcfb')
        print(f"Figure saved to {fig_path}")
    else:
        plt.show()
    plt.close(fig)


def benchmark_sequences(args):
    # SEQ_LENS = [2 ** i for i in range(8, 14)] if seq_lens is None else seq_lens

    SEQ_LENS = args.seq_lens
    batch_size = args.batch
    n_heads = args.heads
    DIM = args.dims

    is_causal = args.causal

    mha_naive = lambda Q, K, V: MultiheadAttentionFunction.apply(
            Q, K, V,
            n_heads * DIM, n_heads, None, None, is_causal, False)
    
    mha_flash = lambda Q, K, V: MultiheadAttentionFunction.apply(
            Q, K, V,
            n_heads * DIM, n_heads, None, None, is_causal, True)
    
    results = pd.DataFrame(columns=('seq_len', 'naive_time', 'flash_time', 'speedup'))
    for seq_len in SEQ_LENS:
        naive_func = run_4d(3, (batch_size, n_heads, seq_len, DIM), mha_naive, torch.device("cuda"))
        flash_func = run_4d(3, (batch_size, n_heads, seq_len, DIM), mha_flash, torch.device("cuda"))

        avg_time_naive = benchmark(naive_func, nvtx_label="naive")
        avg_time_flash = benchmark(flash_func, nvtx_label="flash")
        speedup = avg_time_naive / avg_time_flash

        results.loc[len(results)] = (seq_len, avg_time_naive, avg_time_flash, speedup)

    plot_results(results, args.fig_path)


def benchmark_single(args):
    batch_size = args.batch
    n_heads = args.heads
    N = args.seq_lens
    DIM = args.dims

    is_causal = args.causal
    
    mha_naive = lambda Q, K, V: MultiheadAttentionFunction.apply(
            Q, K, V,
            n_heads * DIM, n_heads, None, None, is_causal, False)
    
    mha_flash = lambda Q, K, V: MultiheadAttentionFunction.apply(
            Q, K, V,
            n_heads * DIM, n_heads, None, None, is_causal, True)

    naive_func = run_4d(3, (batch_size, n_heads, N, DIM), mha_naive, torch.device("cuda"))
    flash_func = run_4d(3, (batch_size, n_heads, N, DIM), mha_flash, torch.device("cuda"))
    
    avg_time_naive = benchmark(naive_func, nvtx_label="naive")
    avg_time_flash = benchmark(flash_func, nvtx_label="flash")

    speedup = avg_time_naive / avg_time_flash
    print(f"Config: batch={batch_size}, heads={n_heads}, seq_len={N}, head_dim={DIM}")
    print(f"  Naive attention : {avg_time_naive:.4f} ms")
    print(f"  Flash attention : {avg_time_flash:.4f} ms")
    print(f"  Speedup         : {speedup:.2f}x")


def run_4d(n_args: int, size: Tuple, func: Callable, device: torch.device | None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = [torch.randn(size, device=device) for _ in range(n_args)]
    return lambda : func(*args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark naive attention vs flash attention")
    parser.add_argument("-b", "--batch", type=int, default=2, help="tensor batch size")
    parser.add_argument("-H", "--heads", type=int, default=8, help="tensor head size")
    parser.add_argument("-s", "--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096], help="tensor sequence lengths")
    parser.add_argument("-d", "--dims", type=int, default=32, help="tensor head dim size")
    parser.add_argument("-c", "--causal", action="store_true", help="If attention is causal i.e. autoregressive")
    parser.add_argument("--fig_path", type=str, default=None, help="Path to save the benchmark figure")
    parser.add_argument("--single", action="store_true", help="Benchmark a single sequence length instead of multiple")

    args = parser.parse_args()
    if args.single:
        benchmark_single(args)
    else:
        benchmark_sequences(args)
