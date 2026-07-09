import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import torch
import torch.nn as nn
from modules.MultiheadAttention import MultiheadAttention, MultiheadAttentionFunction

EMB = 8
HEADS = 2
BATCH = 4
SEQ = 6

# Larger dims needed for flash attn kernel (BLOCK_SIZE >= 16, tl.dot needs power-of-2 dims)
FLASH_EMB = 64
FLASH_HEADS = 4
FLASH_BATCH = 2
FLASH_SEQ = 64

DEVICE = torch.device("cuda:0")


def make_ref(our_layer, embed_dim, num_heads):
    """Build nn.MultiheadAttention with identical weights to our_layer."""
    has_bias = our_layer.proj_q.bias is not None
    ref = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=has_bias)
    with torch.no_grad():
        ref.in_proj_weight[:embed_dim].copy_(our_layer.proj_q.weight)
        ref.in_proj_weight[embed_dim:2*embed_dim].copy_(our_layer.proj_k.weight)
        ref.in_proj_weight[2*embed_dim:].copy_(our_layer.proj_v.weight)
        ref.out_proj.weight.copy_(our_layer.proj_o.weight)
        if has_bias:
            ref.in_proj_bias[:embed_dim].copy_(our_layer.proj_q.bias)
            ref.in_proj_bias[embed_dim:2*embed_dim].copy_(
                our_layer.proj_k.bias if our_layer.proj_k.bias is not None else torch.zeros(embed_dim)
            )
            ref.in_proj_bias[2*embed_dim:].copy_(
                our_layer.proj_v.bias if our_layer.proj_v.bias is not None else torch.zeros(embed_dim)
            )
            ref.out_proj.bias.copy_(our_layer.proj_o.bias)
    return ref


@pytest.fixture
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


# ---------------------------------------------------------------------------
# Forward — output shape
# ---------------------------------------------------------------------------

def test_output_shape_self_attn():
    layer = MultiheadAttention(EMB, HEADS, flash_attn=False)
    x = torch.randn(BATCH, SEQ, EMB)
    assert layer(x, x, x).shape == (BATCH, SEQ, EMB)

def test_output_shape_cross_attn():
    layer = MultiheadAttention(EMB, HEADS, flash_attn=False)
    q = torch.randn(BATCH, SEQ, EMB)
    k = torch.randn(BATCH, SEQ + 2, EMB)
    v = torch.randn(BATCH, SEQ + 2, EMB)
    assert layer(q, k, v).shape == (BATCH, SEQ, EMB)

# ---------------------------------------------------------------------------
# Forward — matches nn.MultiheadAttention
# ---------------------------------------------------------------------------

def _forward_match(bias, add_bias_kv):
    layer = MultiheadAttention(EMB, HEADS, bias=bias, add_bias_kv=add_bias_kv, flash_attn=False)
    ref = make_ref(layer, EMB, HEADS)
    x = torch.randn(BATCH, SEQ, EMB)
    our_out = layer(x, x, x)
    ref_out = ref(x, x, x, need_weights=False)[0]
    assert torch.allclose(our_out, ref_out, atol=1e-5), \
        f"max diff: {(our_out - ref_out).abs().max()}"

def test_forward_no_bias():
    _forward_match(bias=False, add_bias_kv=False)

def test_forward_bias_no_kv_bias():
    _forward_match(bias=True, add_bias_kv=False)

def test_forward_full_bias():
    _forward_match(bias=True, add_bias_kv=True)

# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

def test_causal_mask_changes_output():
    layer = MultiheadAttention(EMB, HEADS, bias=False, flash_attn=False)
    x = torch.randn(BATCH, SEQ, EMB)
    out_causal = layer(x, x, x, is_causal=True)
    out_normal = layer(x, x, x, is_causal=False)
    assert not torch.allclose(out_causal, out_normal), \
        "causal mask should change output"

def test_causal_mask_last_token_differs():
    layer = MultiheadAttention(EMB, HEADS, bias=False, flash_attn=False)
    x = torch.randn(BATCH, SEQ, EMB)
    out_causal = layer(x, x, x, is_causal=True)
    out_normal = layer(x, x, x, is_causal=False)
    assert not torch.allclose(out_causal[:, 1, :], out_normal[:, 1, :], atol=1e-5), \
        "token 1 attends to [0,1] with causal vs all tokens without — should differ"

# ---------------------------------------------------------------------------
# key_padding_mask
# ---------------------------------------------------------------------------

def test_key_padding_mask_changes_output():
    layer = MultiheadAttention(EMB, HEADS, bias=False, flash_attn=False)
    x = torch.randn(BATCH, SEQ, EMB)
    mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
    mask[:, -1] = True
    out_masked = layer(x, x, x, key_padding_mask=mask)
    out_normal = layer(x, x, x)
    assert not torch.allclose(out_masked, out_normal)

def test_key_padding_mask_wrong_shape_raises():
    layer = MultiheadAttention(EMB, HEADS, flash_attn=False)
    x = torch.randn(BATCH, SEQ, EMB)
    bad_mask = torch.zeros(BATCH, SEQ + 1, dtype=torch.bool)
    try:
        layer(x, x, x, key_padding_mask=bad_mask)
        assert False, "should have raised"
    except (KeyError, ValueError):
        pass

# ---------------------------------------------------------------------------
# Backward — gradient match with nn.MultiheadAttention
# ---------------------------------------------------------------------------

def _grad_match(bias, add_bias_kv):
    layer = MultiheadAttention(EMB, HEADS, bias=bias, add_bias_kv=add_bias_kv, flash_attn=False)
    ref = make_ref(layer, EMB, HEADS)

    x1 = torch.randn(BATCH, SEQ, EMB, requires_grad=True)
    x2 = x1.detach().requires_grad_(True)

    layer(x1, x1, x1).sum().backward()
    ref(x2, x2, x2, need_weights=False)[0].sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-4), \
        f"grad_query mismatch, max diff: {(x1.grad - x2.grad).abs().max()}"
    assert torch.allclose(layer.proj_q.weight.grad, ref.in_proj_weight.grad[:EMB], atol=1e-4), \
        "grad_W_q mismatch"
    assert torch.allclose(layer.proj_o.weight.grad, ref.out_proj.weight.grad, atol=1e-4), \
        "grad_W_o mismatch"

def test_grad_no_bias():
    _grad_match(bias=False, add_bias_kv=False)

def test_grad_full_bias():
    _grad_match(bias=True, add_bias_kv=True)

# ---------------------------------------------------------------------------
# gradcheck
# ---------------------------------------------------------------------------

def test_gradcheck_no_bias():
    E, H = 4, 2
    head_dim = E // H
    Q = torch.randn(2, H, 3, head_dim, dtype=torch.float64, requires_grad=True)
    K = torch.randn(2, H, 3, head_dim, dtype=torch.float64, requires_grad=True)
    V = torch.randn(2, H, 3, head_dim, dtype=torch.float64, requires_grad=True)
    result = torch.autograd.gradcheck(
        lambda Q, K, V: MultiheadAttentionFunction.apply(Q, K, V, E, H, None, None, False, False),
        (Q, K, V), eps=1e-6, atol=1e-4
    )
    assert result

def test_gradcheck_with_bias():
    E, H = 4, 2
    layer = MultiheadAttention(E, H, bias=True, add_bias_kv=True, flash_attn=False).double()
    q = torch.randn(2, 3, E, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, E, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, E, dtype=torch.float64, requires_grad=True)
    result = torch.autograd.gradcheck(
        lambda q, k, v: layer(q, k, v),
        (q, k, v), eps=1e-6, atol=1e-4
    )
    assert result

# ---------------------------------------------------------------------------
# Module correctness
# ---------------------------------------------------------------------------

def test_parameters_bias_no_kv():
    layer = MultiheadAttention(EMB, HEADS, bias=True, add_bias_kv=False)
    assert len(list(layer.parameters())) == 6

def test_parameters_full_bias():
    layer = MultiheadAttention(EMB, HEADS, bias=True, add_bias_kv=True)
    assert len(list(layer.parameters())) == 8

def test_parameters_no_bias():
    layer = MultiheadAttention(EMB, HEADS, bias=False, add_bias_kv=False)
    assert len(list(layer.parameters())) == 4

def test_head_count_assertion():
    try:
        MultiheadAttention(embed_dim=9, num_heads=4)
        assert False, "should have raised"
    except AssertionError:
        pass

# ---------------------------------------------------------------------------
# RoPE integration
# ---------------------------------------------------------------------------

def _make_rope_pair(bias=False):
    layer_rope   = MultiheadAttention(EMB, HEADS, rope_seq_len=SEQ * 2, bias=bias, flash_attn=False)
    layer_norope = MultiheadAttention(EMB, HEADS, bias=bias, flash_attn=False)
    with torch.no_grad():
        layer_norope.proj_q.weight.copy_(layer_rope.proj_q.weight)
        layer_norope.proj_k.weight.copy_(layer_rope.proj_k.weight)
        layer_norope.proj_v.weight.copy_(layer_rope.proj_v.weight)
        layer_norope.proj_o.weight.copy_(layer_rope.proj_o.weight)
    return layer_rope, layer_norope


def test_rope_output_shape_self_attn():
    layer = MultiheadAttention(EMB, HEADS, rope_seq_len=SEQ, flash_attn=False)
    x = torch.randn(BATCH, SEQ, EMB)
    assert layer(x, x, x).shape == (BATCH, SEQ, EMB)


def test_rope_output_shape_cross_attn():
    layer = MultiheadAttention(EMB, HEADS, rope_seq_len=SEQ + 2, flash_attn=False)
    q = torch.randn(BATCH, SEQ, EMB)
    k = torch.randn(BATCH, SEQ + 2, EMB)
    v = torch.randn(BATCH, SEQ + 2, EMB)
    assert layer(q, k, v).shape == (BATCH, SEQ, EMB)


def test_rope_changes_output():
    layer_rope, layer_norope = _make_rope_pair()
    x = torch.randn(BATCH, SEQ, EMB)
    assert not torch.allclose(layer_rope(x, x, x), layer_norope(x, x, x))


def test_rope_position_sensitivity():
    layer_rope, layer_norope = _make_rope_pair()
    x = torch.randn(1, SEQ, EMB)
    x[0, 1] = x[0, 0].clone()

    out_rope   = layer_rope(x, x, x)
    out_norope = layer_norope(x, x, x)

    assert torch.allclose(out_norope[0, 0], out_norope[0, 1], atol=1e-5), \
        "without RoPE, identical tokens at different positions must give identical outputs"
    assert not torch.allclose(out_rope[0, 0], out_rope[0, 1], atol=1e-5), \
        "with RoPE, identical tokens at different positions must give different outputs"


def test_rope_no_extra_parameters():
    layer_rope   = MultiheadAttention(EMB, HEADS, rope_seq_len=SEQ, bias=False)
    layer_norope = MultiheadAttention(EMB, HEADS, bias=False)
    assert len(list(layer_rope.parameters())) == len(list(layer_norope.parameters()))


def test_rope_backward_gradients_exist():
    layer = MultiheadAttention(EMB, HEADS, rope_seq_len=SEQ, bias=False, flash_attn=False)
    x = torch.randn(BATCH, SEQ, EMB, requires_grad=True)
    layer(x, x, x).sum().backward()
    assert x.grad is not None
    assert layer.proj_q.weight.grad is not None
    assert layer.proj_o.weight.grad is not None


def test_rope_gradcheck():
    E, H = 4, 2
    layer = MultiheadAttention(E, H, rope_seq_len=8, bias=False, flash_attn=False).double()
    q = torch.randn(2, 3, E, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda q: layer(q, q, q),
        (q,), eps=1e-6, atol=1e-4,
    )


# ---------------------------------------------------------------------------
# Flash attention integration
# ---------------------------------------------------------------------------

def _make_flash_pair(bias=False):
    """Two layers with identical weights: one flash, one standard."""
    flash  = MultiheadAttention(FLASH_EMB, FLASH_HEADS, bias=bias, flash_attn=True,  device=DEVICE)
    std    = MultiheadAttention(FLASH_EMB, FLASH_HEADS, bias=bias, flash_attn=False, device=DEVICE)
    with torch.no_grad():
        std.proj_q.weight.copy_(flash.proj_q.weight)
        std.proj_k.weight.copy_(flash.proj_k.weight)
        std.proj_v.weight.copy_(flash.proj_v.weight)
        std.proj_o.weight.copy_(flash.proj_o.weight)
    return flash, std


def test_flash_output_shape(require_cuda):
    layer = MultiheadAttention(FLASH_EMB, FLASH_HEADS, flash_attn=True, device=DEVICE)
    x = torch.randn(FLASH_BATCH, FLASH_SEQ, FLASH_EMB, device=DEVICE)
    assert layer(x, x, x).shape == (FLASH_BATCH, FLASH_SEQ, FLASH_EMB)


def test_flash_matches_standard(require_cuda):
    flash, std = _make_flash_pair()
    x = torch.randn(FLASH_BATCH, FLASH_SEQ, FLASH_EMB, device=DEVICE)
    out_flash = flash(x, x, x)
    out_std   = std(x, x, x)
    assert torch.allclose(out_flash, out_std, atol=1e-2), \
        f"flash vs standard max diff: {(out_flash - out_std).abs().max():.5f}"


def test_flash_causal_matches_standard(require_cuda):
    flash, std = _make_flash_pair()
    x = torch.randn(FLASH_BATCH, FLASH_SEQ, FLASH_EMB, device=DEVICE)
    out_flash = flash(x, x, x, is_causal=True)
    out_std   = std(x, x, x, is_causal=True)
    assert torch.allclose(out_flash, out_std, atol=1e-2), \
        f"flash causal vs standard causal max diff: {(out_flash - out_std).abs().max():.5f}"


def test_flash_backward_gradients_flow(require_cuda):
    layer = MultiheadAttention(FLASH_EMB, FLASH_HEADS, flash_attn=True, device=DEVICE)
    x = torch.randn(FLASH_BATCH, FLASH_SEQ, FLASH_EMB, device=DEVICE, requires_grad=True)
    layer(x, x, x).sum().backward()
    assert x.grad is not None
    assert layer.proj_q.weight.grad is not None
    assert layer.proj_o.weight.grad is not None


def test_flash_causal_backward_gradients_flow(require_cuda):
    layer = MultiheadAttention(FLASH_EMB, FLASH_HEADS, flash_attn=True, device=DEVICE)
    x = torch.randn(FLASH_BATCH, FLASH_SEQ, FLASH_EMB, device=DEVICE, requires_grad=True)
    layer(x, x, x, is_causal=True).sum().backward()
    assert x.grad is not None
    assert layer.proj_q.weight.grad is not None
    assert layer.proj_o.weight.grad is not None


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
