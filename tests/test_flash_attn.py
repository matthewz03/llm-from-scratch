import math
import pytest
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.kernels.FlashAttention import flash_attn_forward, flash_attn_backward

DEVICE = torch.device("cuda:0")


def _ref(Q, K, V):
    S = Q @ K.T / math.sqrt(Q.shape[-1])
    return F.softmax(S, dim=-1) @ V


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def _qkv(N_Q, N_KV, DIM, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(N_Q, DIM, device=DEVICE)
    K = torch.randn(N_KV, DIM, device=DEVICE)
    V = torch.randn(N_KV, DIM, device=DEVICE)
    return Q, K, V


# ── forward ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("N_Q,N_KV,DIM", [
    (128, 128, 64),   # square, exact multiple of block size
    (256, 128, 64),   # N_Q > N_KV
    (128, 256, 64),   # N_KV > N_Q
])
def test_forward_matches_reference(N_Q, N_KV, DIM):
    Q, K, V = _qkv(N_Q, N_KV, DIM)
    ref = _ref(Q, K, V)
    out, _ = flash_attn_forward(Q, K, V, Q.device)
    assert torch.allclose(ref, out, atol=1e-3), f"max diff: {(ref - out).abs().max():.5f}"


def test_forward_padded_sequence():
    # N not a multiple of BLOCK_SIZE_Q=64 — exercises the tail-block mask
    Q, K, V = _qkv(N_Q=100, N_KV=100, DIM=64)
    ref = _ref(Q, K, V)
    out, _ = flash_attn_forward(Q, K, V, Q.device)
    assert torch.allclose(ref, out, atol=1e-3), f"max diff: {(ref - out).abs().max():.5f}"


def test_forward_output_shape():
    Q, K, V = _qkv(128, 128, 64)
    out, L = flash_attn_forward(Q, K, V, Q.device)
    assert out.shape == Q.shape
    assert L.shape == (Q.shape[0],)


def test_forward_different_seeds():
    # Sanity: different inputs give different outputs
    out1, _ = flash_attn_forward(*_qkv(128, 128, 64, seed=0), DEVICE)
    out2, _ = flash_attn_forward(*_qkv(128, 128, 64, seed=1), DEVICE)
    assert not torch.allclose(out1, out2)


# ── backward ──────────────────────────────────────────────────────────────────

def _ref_grads(Q, K, V, dO):
    """Reference gradients via PyTorch autograd."""
    Q_r = Q.clone().requires_grad_(True)
    K_r = K.clone().requires_grad_(True)
    V_r = V.clone().requires_grad_(True)
    S = Q_r @ K_r.T / math.sqrt(Q.shape[-1])
    O = F.softmax(S, dim=-1) @ V_r
    O.backward(dO)
    return Q_r.grad, K_r.grad, V_r.grad


@pytest.mark.parametrize("N_Q,N_KV,DIM", [
    (128, 128, 64),
    (256, 128, 64),
    (128, 256, 64),
])
def test_backward_matches_reference(N_Q, N_KV, DIM):
    Q, K, V = _qkv(N_Q, N_KV, DIM)
    torch.manual_seed(42)
    dO = torch.randn(N_Q, DIM, device=DEVICE)

    _, L = flash_attn_forward(Q, K, V, DEVICE)
    dQ, dK, dV = flash_attn_backward(Q, K, V, dO, L, DEVICE)

    ref_dQ, ref_dK, ref_dV = _ref_grads(Q, K, V, dO)

    assert torch.allclose(ref_dQ, dQ, atol=1e-2), f"dQ max diff: {(ref_dQ - dQ).abs().max():.5f}"
    assert torch.allclose(ref_dK, dK, atol=1e-2), f"dK max diff: {(ref_dK - dK).abs().max():.5f}"
    assert torch.allclose(ref_dV, dV, atol=1e-2), f"dV max diff: {(ref_dV - dV).abs().max():.5f}"


def test_backward_output_shapes():
    Q, K, V = _qkv(128, 128, 64)
    dO = torch.randn_like(Q)
    _, L = flash_attn_forward(Q, K, V, DEVICE)
    dQ, dK, dV = flash_attn_backward(Q, K, V, dO, L, DEVICE)
    assert dQ.shape == Q.shape
    assert dK.shape == K.shape
    assert dV.shape == V.shape
