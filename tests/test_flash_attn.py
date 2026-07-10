import math
import pytest
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.kernels.FlashAttention import flash_attn_forward, flash_attn_backward

DEVICE = torch.device("cuda:0")
BATCH = 2
HEADS = 4


def _ref(Q, K, V, causal=False):
    S = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
    if causal:
        N_Q, N_KV = Q.shape[-2], K.shape[-2]
        mask = torch.ones(N_Q, N_KV, device=Q.device, dtype=torch.bool).tril()
        S = S.masked_fill(~mask, float('-inf'))
    return F.softmax(S, dim=-1) @ V


def _ref_grads(Q, K, V, dO, causal=False):
    Q_r = Q.clone().requires_grad_(True)
    K_r = K.clone().requires_grad_(True)
    V_r = V.clone().requires_grad_(True)
    S = Q_r @ K_r.transpose(-1, -2) / math.sqrt(Q.shape[-1])
    if causal:
        N_Q, N_KV = Q.shape[-2], K.shape[-2]
        mask = torch.ones(N_Q, N_KV, device=Q.device, dtype=torch.bool).tril()
        S = S.masked_fill(~mask, float('-inf'))
    O = F.softmax(S, dim=-1) @ V_r
    O.backward(dO)
    return Q_r.grad, K_r.grad, V_r.grad


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def _qkv(N_Q, N_KV, DIM, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(BATCH, HEADS, N_Q, DIM, device=DEVICE)
    K = torch.randn(BATCH, HEADS, N_KV, DIM, device=DEVICE)
    V = torch.randn(BATCH, HEADS, N_KV, DIM, device=DEVICE)
    return Q, K, V


# ── forward ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("N_Q,N_KV,DIM", [
    (128, 128, 64),
    (256, 128, 64),
    (128, 256, 64),
])
def test_forward_matches_reference(N_Q, N_KV, DIM):
    Q, K, V = _qkv(N_Q, N_KV, DIM)
    ref = _ref(Q, K, V)
    out, _ = flash_attn_forward(Q, K, V, is_causal=False, device=Q.device)
    assert torch.allclose(ref, out, atol=1e-3), f"max diff: {(ref - out).abs().max():.5f}"


def test_forward_padded_sequence():
    Q, K, V = _qkv(N_Q=100, N_KV=100, DIM=64)
    ref = _ref(Q, K, V)
    out, _ = flash_attn_forward(Q, K, V, is_causal=False, device=Q.device)
    assert torch.allclose(ref, out, atol=1e-3), f"max diff: {(ref - out).abs().max():.5f}"


def test_forward_output_shape():
    Q, K, V = _qkv(128, 128, 64)
    out, L = flash_attn_forward(Q, K, V, is_causal=False, device=Q.device)
    assert out.shape == Q.shape
    assert L.shape == (BATCH, HEADS, Q.shape[-2])


def test_forward_different_seeds():
    out1, _ = flash_attn_forward(*_qkv(128, 128, 64, seed=0), is_causal=False, device=DEVICE)
    out2, _ = flash_attn_forward(*_qkv(128, 128, 64, seed=1), is_causal=False, device=DEVICE)
    assert not torch.allclose(out1, out2)


@pytest.mark.parametrize("N,DIM", [
    (128, 64),
    (256, 64),
])
def test_forward_causal_matches_reference(N, DIM):
    Q, K, V = _qkv(N, N, DIM)
    ref = _ref(Q, K, V, causal=True)
    out, _ = flash_attn_forward(Q, K, V, is_causal=True, device=Q.device)
    assert torch.allclose(ref, out, atol=1e-3), f"max diff: {(ref - out).abs().max():.5f}"


def test_forward_causal_differs_from_noncausal():
    # causal and non-causal should produce different outputs
    Q, K, V = _qkv(128, 128, 64)
    out_causal, _ = flash_attn_forward(Q, K, V, is_causal=True, device=DEVICE)
    out_full, _ = flash_attn_forward(Q, K, V, is_causal=False, device=DEVICE)
    assert not torch.allclose(out_causal, out_full)


# ── backward ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("N_Q,N_KV,DIM", [
    (128, 128, 64),
    (256, 128, 64),
    (128, 256, 64),
])
def test_backward_matches_reference(N_Q, N_KV, DIM):
    Q, K, V = _qkv(N_Q, N_KV, DIM)
    torch.manual_seed(42)
    dO = torch.randn(BATCH, HEADS, N_Q, DIM, device=DEVICE)

    O, L = flash_attn_forward(Q, K, V, is_causal=False, device=DEVICE)
    dQ, dK, dV = flash_attn_backward(Q, K, V, O, L, dO, is_causal=False, device=DEVICE)

    ref_dQ, ref_dK, ref_dV = _ref_grads(Q, K, V, dO)

    assert torch.allclose(ref_dQ, dQ, atol=1e-2), f"dQ max diff: {(ref_dQ - dQ).abs().max():.5f}"
    assert torch.allclose(ref_dK, dK, atol=1e-2), f"dK max diff: {(ref_dK - dK).abs().max():.5f}"
    assert torch.allclose(ref_dV, dV, atol=1e-2), f"dV max diff: {(ref_dV - dV).abs().max():.5f}"


@pytest.mark.parametrize("N,DIM", [
    (128, 64),
    (256, 64),
])
def test_backward_causal_matches_reference(N, DIM):
    Q, K, V = _qkv(N, N, DIM)
    torch.manual_seed(42)
    dO = torch.randn(BATCH, HEADS, N, DIM, device=DEVICE)

    O, L = flash_attn_forward(Q, K, V, is_causal=True, device=DEVICE)
    dQ, dK, dV = flash_attn_backward(Q, K, V, O, L, dO, is_causal=True, device=DEVICE)

    ref_dQ, ref_dK, ref_dV = _ref_grads(Q, K, V, dO, causal=True)

    assert torch.allclose(ref_dQ, dQ, atol=1e-2), f"dQ max diff: {(ref_dQ - dQ).abs().max():.5f}"
    assert torch.allclose(ref_dK, dK, atol=1e-2), f"dK max diff: {(ref_dK - dK).abs().max():.5f}"
    assert torch.allclose(ref_dV, dV, atol=1e-2), f"dV max diff: {(ref_dV - dV).abs().max():.5f}"


def test_backward_output_shapes():
    Q, K, V = _qkv(128, 128, 64)
    dO = torch.randn_like(Q)
    O, L = flash_attn_forward(Q, K, V, is_causal=False, device=DEVICE)
    dQ, dK, dV = flash_attn_backward(Q, K, V, O, L, dO, is_causal=False, device=DEVICE)
    assert dQ.shape == Q.shape
    assert dK.shape == K.shape
    assert dV.shape == V.shape
