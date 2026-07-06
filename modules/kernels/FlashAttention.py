import torch

import triton
import triton.language as tl
from triton.runtime import driver


DEVICE = driver.active.get_active_torch_device()

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def get_cuda_configs():
    return [
        triton.Config(
            {
                'BLOCK_SIZE_Q': 2**q,
                'BLOCK_SIZE_KV': 2**k,
                'num_stages': s
            },
            num_warps=2**w,
        )
        for q in range(4, 6)
            for k in range(4, 6)
                for w in range(1, 4)
                    for s in range(2, 5)
    ]

@triton.autotune(configs=get_cuda_configs(), key=['N_Q', 'N_KV', 'DIM'])
@triton.jit
def flash_attn_forward_kernel(
        q_ptr, 
        k_ptr, 
        v_ptr,
        o_ptr,
        l_ptr,
        q_row_stride,
        q_col_stride,
        k_row_stride,
        k_col_stride,
        v_row_stride,
        v_col_stride,
        o_row_stride,
        o_col_stride,
        l_stride,
        N_Q,
        N_KV,
        DIM: tl.constexpr,
        num_stages: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr
    ):

    pid = tl.program_id(axis=0)

    qo_row_idx = pid * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    qo_col_idx = tl.arange(0, DIM)
    qo_mask = (qo_row_idx < N_Q)[:, None] & (qo_col_idx < DIM)[None, :]
    
    q_row_offsets = qo_row_idx * q_row_stride
    q_col_offsets = qo_col_idx * q_col_stride
    q_offsets = q_row_offsets[:, None] + q_col_offsets[None, :]
    Q = tl.load(q_ptr + q_offsets, mask=qo_mask, other=0.0)

    m = tl.full((BLOCK_SIZE_Q, 1), value=float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_SIZE_Q, 1), dtype=tl.float32)
    O = tl.zeros((BLOCK_SIZE_Q, DIM), dtype=tl.float32)

    d_k = DIM ** (0.5)

    for i in tl.range(0, N_KV, BLOCK_SIZE_KV, num_stages=num_stages):
        kv_row_idx = i + tl.arange(0, BLOCK_SIZE_KV)
        kv_col_idx = tl.arange(0, DIM)
        kv_mask = (kv_row_idx < N_KV)[:, None] & (kv_col_idx < DIM)[None, :]
        
        k_row_offsets = kv_row_idx * k_row_stride
        k_col_offsets = kv_col_idx * k_col_stride
        k_offsets = k_row_offsets[:, None] + k_col_offsets[None, :]
        K = tl.load(k_ptr + k_offsets, mask=kv_mask, other=0.0)
    
        v_row_offsets = kv_row_idx * v_row_stride
        v_col_offsets = kv_col_idx * v_col_stride
        v_offsets = v_row_offsets[:, None] + v_col_offsets[None, :]
        V = tl.load(v_ptr + v_offsets, mask=kv_mask, other=0.0)

        S = tl.dot(Q, tl.trans(K)) / d_k
        bound_mask = (kv_row_idx < N_KV)
        S = tl.where(bound_mask[None, :], S, float('-inf'))

        curr_m = tl.max(S, axis=1, keep_dims=True)
        new_m = tl.maximum(m, curr_m)

        e_diff = tl.exp(m - new_m) # e^(m_i-1 - m)
        scaled_S = tl.exp(S - new_m) # e^(S_i - m) ∈ (BLOCK_Q, BLOCK_KV)
        l = e_diff * l + tl.sum(scaled_S, axis=1, keep_dims=True)

        O = e_diff * O + tl.dot(scaled_S, V)
        m = new_m
    
    O /= l
    o_row_offsets = qo_row_idx * o_row_stride
    o_col_offsets = qo_col_idx * o_col_stride
    o_offsets = o_row_offsets[:, None] + o_col_offsets[None, :]
    
    L = m + tl.log(l)
    l_offsets = qo_row_idx * l_stride
    l_mask = qo_row_idx < N_Q
    
    tl.store(o_ptr + o_offsets, O, mask=qo_mask)
    tl.store(l_ptr + l_offsets, tl.reshape(L, [BLOCK_SIZE_Q]), mask=l_mask)

def flash_attn_forward(
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        device: torch.device
    ):
    
    assert Q.device == K.device == V.device == device, 'Q, K, V must be the same device'
    assert K.shape == V.shape, 'K, V must be the same shape'
    assert Q.shape[-1] == K.shape[-1], 'Q, K, V must have the same feature dim'
    assert len(Q.shape) == len(K.shape)

    O = torch.empty_like(Q, dtype=Q.dtype, device=device)
    N_Q = Q.shape[-2]
    N_KV = K.shape[-2]
    DIM = Q.shape[-1]
    L = torch.empty(N_Q, dtype=Q.dtype, device=device)

    grid = lambda meta: (triton.cdiv(N_Q, meta['BLOCK_SIZE_Q']), )

    flash_attn_forward_kernel[grid](
        Q, 
        K, 
        V, 
        O,
        L,
        Q.stride(-2),
        Q.stride(-1),
        K.stride(-2),
        K.stride(-1),
        V.stride(-2),
        V.stride(-1),
        O.stride(-2),
        O.stride(-1),
        L.stride(-1),
        N_Q=N_Q,
        N_KV=N_KV,
        DIM=DIM,
    )

    return O, L

@triton.autotune(configs=get_cuda_configs(), key=['N_Q', 'N_KV', 'DIM'])
@triton.jit
def flash_attn_backward_kernel(
        q_ptr, 
        k_ptr, 
        v_ptr,
        do_ptr,
        d_ptr,
        l_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        q_row_stride,
        q_col_stride,
        k_row_stride,
        k_col_stride,
        v_row_stride,
        v_col_stride,
        do_row_stride,
        do_col_stride,
        dq_row_stride,
        dq_col_stride,
        dk_row_stride,
        dk_col_stride,
        dv_row_stride,
        dv_col_stride,
        d_stride,
        l_stride, # l.shape is (N_Q,)
        N_Q,
        N_KV,
        DIM: tl.constexpr,
        num_stages: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr
    ):
    '''
    Keep m, l from forward pass, invert the inner and outer loop
    go through a column of Q, fixed K,V for each block
    Given: 
        dO ∈ (N_Q, DIM)
        Q ∈ (N_Q, DIM)
        K ∈ (N_KV, DIM)
        V ∈ (N_KV, DIM)
        m ∈ (N_Q, )
        l ∈ (N_Q, )
         
    Calculate:
        P ∈ (N_Q, N_KV) - post softmax attn matrix
        S ∈ (N_Q, N_KV) - pre softmax attn matrix
        
        dQ ∈ (N_Q, DIM)
        dK ∈ (N_KV, DIM)
        dV ∈ (N_KV, DIM)
    
    dV = P.T @ dO
    dP = dO @ V.T
    dS = d_softmax(P)
    dQ = dS @ K
    dK = dS.T @ Q

    Blocks:
        dO ∈ (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
    '''

    pid = tl.program_id(axis=0)

    kv_row_idx = pid * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)
    kv_col_idx = tl.arange(0, DIM)
    kv_mask = (kv_row_idx < N_KV)[:, None] & (kv_col_idx < DIM)[None, :]

    k_row_offsets = kv_row_idx * k_row_stride
    k_col_offsets = kv_col_idx * k_col_stride
    k_offsets = k_row_offsets[:, None] + k_col_offsets[None, :]
    K = tl.load(k_ptr + k_offsets, mask=kv_mask)

    v_row_offsets = kv_row_idx * v_row_stride
    v_col_offsets = kv_col_idx * v_col_stride
    v_offsets = v_row_offsets[:, None] + v_col_offsets[None, :]
    V = tl.load(v_ptr + v_offsets, mask=kv_mask)


    dV = tl.zeros_like(V)
    dK = tl.zeros_like(K)
    
    d_k = DIM ** (0.5)

    for i in tl.range(0, N_Q, BLOCK_SIZE_Q, num_stages=num_stages):

        qo_row_idx = i + tl.arange(0, BLOCK_SIZE_Q)
        qo_col_idx = tl.arange(0, DIM)
        qo_mask = (qo_row_idx < N_Q)[:, None] & (qo_col_idx < DIM)[None, :]
        
        q_row_offsets = qo_row_idx * q_row_stride
        q_col_offsets = qo_col_idx * q_col_stride
        q_offsets = q_row_offsets[:, None] + q_col_offsets[None, :]
        Q = tl.load(q_ptr + q_offsets, mask=qo_mask, other=0.0)

        do_row_offsets = qo_row_idx * do_row_stride
        do_col_offsets = qo_col_idx * do_col_stride
        do_offsets = do_row_offsets[:, None] + do_col_offsets[None, :]
        dO = tl.load(do_ptr + do_offsets, mask=qo_mask, other=0.0)

        # m_offsets = qo_row_idx * m_stride
        # m = tl.load(m_ptr + m_offsets, mask=ml_mask)
        # m = m[:, None]

        l_offsets = qo_row_idx * l_stride
        d_offsets = qo_row_idx * d_stride
        ld_mask = qo_row_idx < N_Q

        L = tl.load(l_ptr + l_offsets, mask=ld_mask)
        L = L[:, None]
        D = tl.load(d_ptr + d_offsets, mask=ld_mask)
        D = D[:, None]

        # L = m + ln(sum)
        # exp(s - L) = exp(s - (m + ln(sum)) = exp(s - m) / exp(ln(sum)) = exp(s - m) / sum 
        S = tl.dot(Q, tl.trans(K)) / d_k
        P = tl.exp(S - L)

        dV += tl.dot(tl.trans(P), dO) # (BLOCK_SIZE_KV, BLOCK_SIZE_Q) X (BLOCK_SIZE_Q, DIM)
        dP = tl.dot(dO, tl.trans(V))

        # softmax derivative:
        #   s = exp(x - m) / s
        #   dS = softmax(S) * dx + softmax(S) * dm + softmax(S) * ds
        #   softmax(S) * dx = exp(x - m) / s = S
        #   softmax(S) * ds = -exp(x - m) / s^2 * dx 
        #                   = sum(-exp(x - m) / s^2, dim=-1) * exp(x - m)
        #                   = -sum(P, dim=-1) * P

        # dS - (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
        dS = P * (dP - D) / d_k
        dK += tl.dot(tl.trans(dS), Q)
        dQ = tl.dot(dS, K) # (BLOCK_SIZE_Q, DIM)
        
        dq_row_offsets = qo_row_idx * dq_row_stride
        dq_col_offsets = qo_col_idx * dq_col_stride
        dq_offsets = dq_row_offsets[:, None] + dq_col_offsets[None, :]

        tl.atomic_add(dq_ptr + dq_offsets, dQ, mask=qo_mask)

    dk_row_offsets = kv_row_idx * dk_row_stride
    dk_col_offsets = kv_col_idx * dk_col_stride
    dk_offsets = dk_row_offsets[:, None] + dk_col_offsets[None, :]

    dv_row_offsets = kv_row_idx * dv_row_stride
    dv_col_offsets = kv_col_idx * dv_col_stride
    dv_offsets = dv_row_offsets[:, None] + dv_col_offsets[None, :]

    tl.store(dk_ptr + dk_offsets, dK, mask=kv_mask)
    tl.store(dv_ptr + dv_offsets, dV, mask=kv_mask)

def flash_attn_backward(
        Q, 
        K, 
        V, 
        O,
        grad_upstream, 
        L, 
        device: torch.device
    ):

    assert (
        Q.device == 
        K.device == 
        V.device == 
        O.device == 
        grad_upstream.device == 
        L.device == device
    ), 'Q, K, V must be the same device'
    assert K.shape == V.shape, 'K, V must be the same shape'
    assert Q.shape == grad_upstream.shape == O.shape, 'Q, grad_upstream, O must have the same shape'
    assert Q.shape[-1] == K.shape[-1], 'Q, K, V, grad_upstream must have the same feature dim'
    assert len(Q.shape) == len(K.shape), 'Q, K, V must have the same number of dims'
    assert len(L.shape) == 1, 'L must be 1 dimensional'
    assert Q.shape[-2] == L.shape[0]

    grad_Q, grad_K, grad_V = torch.zeros_like(Q, device=device), torch.empty_like(K, device=device), torch.empty_like(V, device=device)
    D = (grad_upstream * O).sum(dim=-1)
    N_Q, N_KV = Q.shape[-2], K.shape[-2]

    grid = lambda meta: (triton.cdiv(N_KV, meta['BLOCK_SIZE_KV']), )

    flash_attn_backward_kernel[grid](
        Q,
        K,
        V,
        grad_upstream,
        D,
        L,
        grad_Q,
        grad_K,
        grad_V,
        Q.stride(-2),
        Q.stride(-1),
        K.stride(-2),
        K.stride(-1),
        V.stride(-2),
        V.stride(-1),
        grad_upstream.stride(-2),
        grad_upstream.stride(-1),
        grad_Q.stride(-2),
        grad_Q.stride(-1),
        grad_K.stride(-2),
        grad_K.stride(-1),
        grad_V.stride(-2),
        grad_V.stride(-1),
        D.stride(-1),
        L.stride(-1),
        N_Q=N_Q,
        N_KV=N_KV,
        DIM=Q.shape[-1],
    )

    return grad_Q, grad_K, grad_V