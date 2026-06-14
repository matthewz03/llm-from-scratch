import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from modules.RotaryPositionalEmbeddings import RotaryPositionEmbeddings

DIM = 8
MAX_SEQ = 16
BATCH = 2
SEQ = 6


def make_rope(dim=DIM, max_seq=MAX_SEQ, base=10000):
    return RotaryPositionEmbeddings(dim=dim, max_seq_len=max_seq, base=base)


def ref_apply_rope(x, dim, base, seq_len):
    """Naive reference: explicit rotation formula applied to each dimension pair."""
    theta = base ** (-2 * torch.arange(dim // 2).float() / dim)
    m_theta = torch.arange(seq_len).float().outer(theta)  # (seq_len, dim//2)
    cos_t = m_theta.cos().unsqueeze(0)   # (1, seq_len, dim//2)
    sin_t = m_theta.sin().unsqueeze(0)
    x0, x1 = x[..., ::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2]  = x0 * cos_t - x1 * sin_t
    out[..., 1::2] = x0 * sin_t + x1 * cos_t
    return out


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_output_shape():
    rope = make_rope()
    q = torch.randn(BATCH, SEQ, DIM)
    k = torch.randn(BATCH, SEQ, DIM)
    q_out, k_out = rope(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_position_zero_is_identity():
    """At position 0, all angles are 0: cos=1, sin=0, so output == input."""
    rope = make_rope()
    q = torch.randn(BATCH, SEQ, DIM)
    k = torch.randn(BATCH, SEQ, DIM)
    q_out, k_out = rope(q, k)
    assert torch.allclose(q_out[:, 0, :], q[:, 0, :], atol=1e-6), "position 0 should be identity"
    assert torch.allclose(k_out[:, 0, :], k[:, 0, :], atol=1e-6), "position 0 should be identity"


def test_matches_reference():
    """Output matches naive explicit rotation formula."""
    rope = make_rope()
    q = torch.randn(BATCH, SEQ, DIM)
    k = torch.randn(BATCH, SEQ, DIM)
    q_out, k_out = rope(q, k)
    q_ref = ref_apply_rope(q, DIM, 10000, SEQ)
    k_ref = ref_apply_rope(k, DIM, 10000, SEQ)
    assert torch.allclose(q_out, q_ref, atol=1e-5), f"q max diff: {(q_out - q_ref).abs().max()}"
    assert torch.allclose(k_out, k_ref, atol=1e-5), f"k max diff: {(k_out - k_ref).abs().max()}"


def test_rotation_preserves_norm():
    """RoPE is an orthogonal rotation — must preserve the L2 norm of each token vector."""
    rope = make_rope()
    q = torch.randn(BATCH, SEQ, DIM)
    k = torch.randn(BATCH, SEQ, DIM)
    q_out, k_out = rope(q, k)
    assert torch.allclose(q.norm(dim=-1), q_out.norm(dim=-1), atol=1e-5), "q norm changed"
    assert torch.allclose(k.norm(dim=-1), k_out.norm(dim=-1), atol=1e-5), "k norm changed"


def test_different_positions_differ():
    """A uniform input rotated at different positions should yield distinct outputs."""
    rope = make_rope()
    q = torch.ones(1, 4, DIM)
    q_out, _ = rope(q, q)
    assert not torch.allclose(q_out[:, 0, :], q_out[:, 1, :]), \
        "tokens at different positions should have different rotations"


def test_seq_len_slicing_consistency():
    """Rotating the first k tokens of a long sequence == rotating a k-token sequence."""
    rope = make_rope()
    q = torch.randn(BATCH, 8, DIM)
    k = torch.randn(BATCH, 8, DIM)
    q_long, k_long = rope(q, k)
    q_short, k_short = rope(q[:, :4, :], k[:, :4, :])
    assert torch.allclose(q_long[:, :4, :], q_short, atol=1e-6)
    assert torch.allclose(k_long[:, :4, :], k_short, atol=1e-6)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

def test_grad_flows_through_query():
    rope = make_rope()
    q = torch.randn(BATCH, SEQ, DIM, requires_grad=True)
    k = torch.randn(BATCH, SEQ, DIM)
    q_out, _ = rope(q, k)
    q_out.sum().backward()
    assert q.grad is not None and q.grad.shape == q.shape


def test_grad_flows_through_key():
    rope = make_rope()
    q = torch.randn(BATCH, SEQ, DIM)
    k = torch.randn(BATCH, SEQ, DIM, requires_grad=True)
    _, k_out = rope(q, k)
    k_out.sum().backward()
    assert k.grad is not None and k.grad.shape == k.shape


def test_gradcheck():
    rope = make_rope(dim=4, max_seq=8).double()
    q = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)

    def fn(q, k):
        q_out, k_out = rope(q, k)
        return torch.cat([q_out, k_out], dim=-1)

    assert torch.autograd.gradcheck(fn, (q, k), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# Module correctness
# ---------------------------------------------------------------------------

def test_no_learnable_parameters():
    assert len(list(make_rope().parameters())) == 0


def test_buffers_registered():
    buf_names = {name for name, _ in make_rope().named_buffers()}
    assert 'cos' in buf_names and 'sin' in buf_names


def test_buffers_follow_dtype():
    rope = make_rope().double()
    assert rope.cos.dtype == torch.float64
    assert rope.sin.dtype == torch.float64


def test_buffer_shape():
    rope = make_rope(dim=DIM, max_seq=MAX_SEQ)
    assert rope.cos.shape == (1, MAX_SEQ, DIM)
    assert rope.sin.shape == (1, MAX_SEQ, DIM)


if __name__ == "__main__":
    import sys
    tests = {k: v for k, v in globals().items() if k.startswith("test_")}
    failed = []
    for name, fn in tests.items():
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(len(failed))
