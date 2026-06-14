import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn as nn
from modules.LayerNorm import LayerNorm, LayerNormFunction


def make_ref(normalized_shape, bias, our_layer):
    ref = nn.LayerNorm(normalized_shape, elementwise_affine=True, bias=bias)
    with torch.no_grad():
        ref.weight.copy_(our_layer.gamma)
        if bias:
            ref.bias.copy_(our_layer.beta)
    return ref


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_forward_2d():
    layer = LayerNorm(8)
    ref = make_ref(8, True, layer)
    x = torch.randn(16, 8)
    assert torch.allclose(layer(x), ref(x), atol=1e-5)

def test_forward_3d():
    layer = LayerNorm(8)
    ref = make_ref(8, True, layer)
    x = torch.randn(4, 10, 8)
    assert torch.allclose(layer(x), ref(x), atol=1e-5)

def test_forward_multi_dim_normalized_shape():
    layer = LayerNorm([10, 8])
    ref = make_ref([10, 8], True, layer)
    x = torch.randn(4, 10, 8)
    assert torch.allclose(layer(x), ref(x), atol=1e-5)

def test_forward_no_bias():
    layer = LayerNorm(8, bias=False)
    ref = make_ref(8, False, layer)
    x = torch.randn(16, 8)
    assert torch.allclose(layer(x), ref(x), atol=1e-5)

def test_forward_int_normalized_shape():
    layer = LayerNorm(8)
    ref = make_ref(8, True, layer)
    x = torch.randn(4, 8)
    assert torch.allclose(layer(x), ref(x), atol=1e-5)

def test_normalized_output_stats():
    layer = LayerNorm(16, bias=False)
    nn.init.ones_(layer.gamma)
    x = torch.randn(32, 16) * 5 + 3
    y = layer(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(32), atol=1e-5)
    assert torch.allclose(y.var(dim=-1, correction=0), torch.ones(32), atol=1e-5)

def test_wrong_shape_raises():
    layer = LayerNorm(8)
    try:
        layer(torch.randn(4, 4))
        assert False, "should have raised"
    except AssertionError:
        pass

# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def _grad_match(normalized_shape, bias, input_shape):
    layer = LayerNorm(normalized_shape, bias=bias)
    ref = make_ref(normalized_shape, bias, layer)

    x1 = torch.randn(*input_shape, requires_grad=True)
    x2 = x1.detach().requires_grad_(True)

    layer(x1).sum().backward()
    ref(x2).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5), "grad_input mismatch"
    assert torch.allclose(layer.gamma.grad, ref.weight.grad, atol=1e-5), "grad_gamma mismatch"
    if bias:
        assert torch.allclose(layer.beta.grad, ref.bias.grad, atol=1e-5), "grad_beta mismatch"

def test_grad_2d_bias():
    _grad_match(8, True, (16, 8))

def test_grad_2d_no_bias():
    _grad_match(8, False, (16, 8))

def test_grad_3d_bias():
    _grad_match(8, True, (4, 10, 8))

def test_grad_3d_no_bias():
    _grad_match(8, False, (4, 10, 8))

def test_grad_multi_dim_normalized_shape():
    _grad_match([10, 8], True, (4, 10, 8))

# ---------------------------------------------------------------------------
# gradcheck
# ---------------------------------------------------------------------------

def test_gradcheck_2d():
    layer = LayerNorm(4).double()
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    w = layer.gamma.detach().requires_grad_(True)
    b = layer.beta.detach().requires_grad_(True)
    assert torch.autograd.gradcheck(LayerNormFunction.apply, (x, w, b, [4], 1e-5), eps=1e-6, atol=1e-4)

def test_gradcheck_3d():
    layer = LayerNorm(4).double()
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    w = layer.gamma.detach().requires_grad_(True)
    b = layer.beta.detach().requires_grad_(True)
    assert torch.autograd.gradcheck(LayerNormFunction.apply, (x, w, b, [4], 1e-5), eps=1e-6, atol=1e-4)

def test_gradcheck_multi_dim():
    layer = LayerNorm([3, 4]).double()
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    w = layer.gamma.detach().requires_grad_(True)
    b = layer.beta.detach().requires_grad_(True)
    assert torch.autograd.gradcheck(LayerNormFunction.apply, (x, w, b, [3, 4], 1e-5), eps=1e-6, atol=1e-4)

# ---------------------------------------------------------------------------
# Module correctness
# ---------------------------------------------------------------------------

def test_parameters_with_bias():
    assert len(list(LayerNorm(8, bias=True).parameters())) == 2

def test_parameters_no_bias():
    assert len(list(LayerNorm(8, bias=False).parameters())) == 1

def test_gamma_shape():
    assert LayerNorm(8).gamma.shape == (8,)
    assert LayerNorm([4, 8]).gamma.shape == (4, 8)

def test_gamma_init_ones():
    assert LayerNorm(8).gamma.allclose(torch.ones(8))

def test_beta_init_zeros():
    assert LayerNorm(8).beta.allclose(torch.zeros(8))


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
