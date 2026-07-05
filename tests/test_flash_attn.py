import math
import pytest
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.kernels.FlashAttention import flash_attn_forward

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
