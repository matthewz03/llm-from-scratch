import torch

import triton
import triton.language as tl
from triton.runtime import driver


def persistent_matmul_kernel(
        x_ptr, 
        y_ptr, 
        out_ptr, 
        x_row_stride, 
        x_col_stride, 
        y_row_stride,
        y_col_stride, 
        out_row_stride,
        out_col_stride,
        I,
        J,
        K, 
        num_stages,
        BLOCK_ROW_SIZE: tl.constexpr,
        BLOCK_COL_SIZE: tl.constexpr, 
        ):
    '''
    O = X @ Y; x.shape = (i, j); y.shape = (j, k)
    shared dim is j, block size will be next power of 2 of j
    number of total blocks needed is i*j
    will need to double loop x, y
    each block will have its own double loop where it will 
        do the matmul and store in out block index position  
    '''

    # which block idx is this
    row_pid, col_pid = tl.program_id(axis=0), tl.program_id(1)
    # what size step: height/width of block * number of blocks
    row_step, col_step = tl.num_programs(axis=0) * BLOCK_ROW_SIZE, tl.num_programs(axis=1) * BLOCK_COL_SIZE

    for i in tl.range(row_pid, I, row_step, num_stages=num_stages):
        x_row_idx = (i + tl.arange(0, BLOCK_ROW_SIZE))
        x_col_idx = tl.arange(0, J)
        x_row_offsets = x_row_stride * x_row_idx
        x_col_offsets = x_col_stride * x_col_idx
        x_offsets = x_row_offsets[:, None] + x_col_offsets[None, :]
        x_mask = (x_row_idx < I)[:, None] & (x_col_idx < J)[None, :]
        X = tl.load(x_ptr + x_offsets, mask=x_mask)

        O = tl.zeros((BLOCK_ROW_SIZE, K), dtype=tl.float32)

        for k in tl.range(col_pid, K, col_step, num_stages=num_stages):
            
            y_row_idx = k + tl.arange(0, J)
            y_col_idx = tl.arange(0, BLOCK_COL_SIZE)
            y_row_offsets = y_row_stride * y_row_idx
            y_col_offsets = y_col_stride * y_col_idx
            y_offsets = y_row_offsets[:, None] + y_col_offsets[None, :]
            y_mask = (y_row_idx < J)[:, None] & (y_col_idx < K)[None, :]
            Y = tl.load(y_ptr + y_offsets, mask=y_mask)

            O[:, k:k+BLOCK_COL_SIZE] = tl.sum(X * tl.trans(Y, (1, 0)), axis=1)
        
        out_row_idx = i + tl.arange(0, BLOCK_ROW_SIZE)
        out_col_idx = tl.arange(0, K)
        out_row_offsets = out_row_stride * out_row_idx
        out_col_offsets = out_col_stride * out_col_idx
        out_offsets = out_row_offsets[:, None] * out_col_offsets[None, :]
        out_mask = (out_row_idx < I)[:, None] * (out_col_idx < K)[None, :]

        tl.store(out_ptr + out_offsets, O, mask=out_mask)

@triton.jit
def matmul_kernel(
        x_ptr, 
        y_ptr, 
        out_ptr, 
        x_stride_row, 
        x_stride_col, 
        y_stride_row,
        y_stride_col,
        out_stride_row,
        out_stride_col,
        I,
        J,
        K, 
        BLOCK_SIZE_I: tl.constexpr,
        BLOCK_SIZE_J: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr, 
        num_stages: tl.constexpr,
        ):
    '''
    O = X @ Y; x.shape = (i, j); y.shape = (j, k)
    shared dim is j, block size will be next power of 2 of j
    number of total blocks needed is i*j
    will need to double loop x, y
    each block will have its own double loop where it will 
        do the matmul and store in out block index position  
    '''

    # which block idx is this
    row_pid, col_pid = tl.program_id(axis=0), tl.program_id(1)
    # what size step: height/width of block * number of blocks

    O = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_K), dtype=tl.float32)


    x_row_idx = row_pid * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    y_col_idx = col_pid * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    for j in tl.range(0, J, BLOCK_SIZE_J, num_stages=num_stages):
        x_col_idx = j + tl.arange(0, BLOCK_SIZE_J)
        x_row_offsets = x_stride_row * x_row_idx
        x_col_offsets = x_stride_col * x_col_idx
        x_offsets = x_row_offsets[:, None] + x_col_offsets[None, :]
        x_mask = (x_row_idx < I)[:, None] & (x_col_idx < J)[None, :]
        X = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        
        y_row_idx = j + tl.arange(0, BLOCK_SIZE_J)
        y_row_offsets = y_stride_row * y_row_idx
        y_col_offsets = y_stride_col * y_col_idx
        y_offsets = y_row_offsets[:, None] + y_col_offsets[None, :]
        y_mask = (y_row_idx < J)[:, None] & (y_col_idx < K)[None, :]
        Y = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)

        O += tl.dot(X, Y)
    
    out_row_offsets = out_stride_row * x_row_idx
    out_col_offsets = out_stride_col * y_col_idx
    out_offsets = out_row_offsets[:, None] + out_col_offsets[None, :]
    out_mask = (x_row_idx < I)[:, None] & (y_col_idx < K)[None, :]

    tl.store(out_ptr + out_offsets, O, mask=out_mask)

def matmul(x: torch.Tensor, y: torch.Tensor):
    assert len(x.shape) == len(y.shape) == 2, "x, y must be 2D matrices"
    assert x.shape[1] == y.shape[0], "Col dimension of x must match row dimension of y"


    I, J, K = x.shape[0], x.shape[1], y.shape[1]
    out = torch.empty(I, K, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(I, meta['BLOCK_SIZE_I']), triton.cdiv(K, meta['BLOCK_SIZE_K']))

    num_stages = 4 if SIZE_SMEM > 200000 else 2
    matmul_kernel[grid](
        x, 
        y, 
        out, 
        x.stride(0), 
        x.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
        I,
        J,
        K, 
        BLOCK_SIZE_I=32,
        BLOCK_SIZE_J=64, 
        BLOCK_SIZE_K=32, 
        num_stages=num_stages,
    )
    
    return out

DEVICE = driver.active.get_active_torch_device()

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

rows, cols = 5000, 2000

X = torch.randn((rows, cols), device=DEVICE)
Y = torch.randn((cols, rows), device=DEVICE)

torch_result = X @ Y
my_result = matmul(X, Y)

assert torch_result == my_result