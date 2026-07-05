import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


@triton.jit
def fused_softmax_vector(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    '''
    softmax(x) = exp(x - max(x)) / sum(x - max(x))

    exp(x - max(x)) * exp(max(x) - curr_max(x)) = exp(x - next_max(x))
    '''
    pid = tl.program_id(axis=0)

    # total = tl.zeros([BLOCK_SIZE])
    max = float('-inf')
    sum = 0.0
    # sum = tl.tensor(0.0, type=tl.float32)

    for i in range(0, N, BLOCK_SIZE):
        # block_start = i * BLOCK_SIZE
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        # e_x = tl.exp(x)
        # tl.store(out_ptr + offsets, e_x, mask=mask)

        curr_max = tl.maximum(tl.max(x), max)
        # curr_sum = tl.sum(e_x * tl.exp(-curr_max))
        curr_sum = tl.sum(tl.exp(x-curr_max))

        max_diff = max - curr_max
        e_diff = tl.exp(max_diff)

        sum = sum * e_diff + curr_sum
        max = curr_max

    for j in range(0, N, BLOCK_SIZE):
        offsets = j + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        # out = tl.load(out_ptr + offsets, mask=mask)
        out = tl.exp(x-max) / sum
        # out = out * tl.exp(-max) / sum
        tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def fused_softmax_matrix(
        x_ptr, 
        out_ptr,
        x_stride, 
        out_stride,
        N_ROWS, 
        N_COLS, 
        BLOCK_SIZE: tl.constexpr,
        num_stages: tl.constexpr, 
    ):
    '''
    softmax(x) = exp(x - max(x)) / sum(x - max(x))

    exp(x - max(x)) * exp(max(x) - curr_max(x)) = exp(x - next_max(x)) 
    '''
    row_pid = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    for idx in tl.range(row_pid, N_ROWS, row_step, num_stages=num_stages):
        row_start = idx * x_stride
        col_offsets = tl.arange(0, BLOCK_SIZE) 
        x_offsets = row_start + col_offsets

        mask = col_offsets < N_COLS

        x = tl.load(x_ptr + x_offsets, mask=mask, other=float('-inf'))

        x_shift = x - tl.max(x, axis=0)
        e_x = tl.exp(x_shift)
        sum_e_x = tl.sum(e_x, axis=0)
        softmax = e_x / sum_e_x

        out_start = idx * out_stride
        out_offsets = out_start + col_offsets

        tl.store(out_ptr + out_offsets, softmax, mask=mask)


properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = fused_softmax_matrix.warmup(
        y, x, x.stride(0), y.stride(0), n_rows, 
        n_cols, BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages, num_warps=num_warps, grid=(1, )
        )
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        # number of Registers / registers used per block
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps) # registers/thread * threads/warp * warps
    occupancy = min(occupancy, SIZE_SMEM // size_smem) # per SM, limited either by registers or SMEM
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y

  
  
