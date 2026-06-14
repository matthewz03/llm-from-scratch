import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import math
import torch
import torch.nn as nn
from modules.Linear import Linear


def make_ref(in_f, out_f, bias, our_linear):
    """Build an nn.Linear with identical weights to our_linear."""
    ref = nn.Linear(in_f, out_f, bias=bias)
    with torch.no_grad():
        ref.weight.copy_(our_linear.weight)
        if bias:
            ref.bias.copy_(our_linear.bias)
    return ref


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_forward_2d_bias():
    layer = Linear(4, 8)
    ref = make_ref(4, 8, True, layer)
    x = torch.randn(16, 4)
    assert torch.allclose(layer(x), ref(x), atol=1e-6)

def test_forward_2d_no_bias():
    layer = Linear(4, 8, bias=False)
    ref = make_ref(4, 8, False, layer)
    x = torch.randn(16, 4)
    assert torch.allclose(layer(x), ref(x), atol=1e-6)

def test_forward_3d_bias():
    layer = Linear(4, 8)
    ref = make_ref(4, 8, True, layer)
    x = torch.randn(16, 10, 4)
    assert torch.allclose(layer(x), ref(x), atol=1e-6)

def test_forward_3d_no_bias():
    layer = Linear(4, 8, bias=False)
    ref = make_ref(4, 8, False, layer)
    x = torch.randn(16, 10, 4)
    assert torch.allclose(layer(x), ref(x), atol=1e-6)

def test_output_shape():
    layer = Linear(4, 8)
    assert layer(torch.randn(16, 4)).shape == (16, 8)
    assert layer(torch.randn(16, 10, 4)).shape == (16, 10, 8)

# ---------------------------------------------------------------------------
# Gradients: match nn.Linear
# ---------------------------------------------------------------------------

def _grad_match(in_f, out_f, bias, input_shape):
    layer = Linear(in_f, out_f, bias=bias)
    ref = make_ref(in_f, out_f, bias, layer)

    x1 = torch.randn(*input_shape, requires_grad=True)
    x2 = x1.detach().requires_grad_(True)

    layer(x1).sum().backward()
    ref(x2).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-6), "grad_input mismatch"
    assert torch.allclose(layer.weight.grad, ref.weight.grad, atol=1e-6), "grad_weight mismatch"
    if bias:
        assert torch.allclose(layer.bias.grad, ref.bias.grad, atol=1e-6), "grad_bias mismatch"

def test_grad_2d_bias():
    _grad_match(4, 8, True, (16, 4))

def test_grad_2d_no_bias():
    _grad_match(4, 8, False, (16, 4))

def test_grad_3d_bias():
    _grad_match(4, 8, True, (16, 10, 4))

def test_grad_3d_no_bias():
    _grad_match(4, 8, False, (16, 10, 4))

# ---------------------------------------------------------------------------
# gradcheck: numerically verifies the analytic backward
# ---------------------------------------------------------------------------

def _gradcheck(bias, input_shape):
    layer = Linear(4, 8, bias=bias).double()
    x = torch.randn(*input_shape, dtype=torch.float64, requires_grad=True)
    bias_tensor = layer.bias if bias else None
    result = torch.autograd.gradcheck(
        lambda inp: torch.nn.functional.linear(inp, layer.weight, bias_tensor),
        (x,),
        eps=1e-6,
        atol=1e-4,
    )
    assert result

def test_gradcheck_2d_bias():
    _gradcheck(True, (4, 4))

def test_gradcheck_2d_no_bias():
    _gradcheck(False, (4, 4))

def test_gradcheck_3d_bias():
    _gradcheck(True, (3, 5, 4))

# ---------------------------------------------------------------------------
# gradcheck through LinearFunction directly
# ---------------------------------------------------------------------------

def test_gradcheck_custom_function():
    from modules.Linear import LinearFunction
    layer = Linear(4, 8).double()
    x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    w = layer.weight.detach().requires_grad_(True)
    b = layer.bias.detach().requires_grad_(True)
    assert torch.autograd.gradcheck(LinearFunction.apply, (x, w, b), eps=1e-6, atol=1e-4)

def test_gradcheck_custom_function_no_bias():
    from modules.Linear import LinearFunction
    layer = Linear(4, 8, bias=False).double()
    x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    w = layer.weight.detach().requires_grad_(True)
    assert torch.autograd.gradcheck(LinearFunction.apply, (x, w, None), eps=1e-6, atol=1e-4)

# ---------------------------------------------------------------------------
# Module correctness
# ---------------------------------------------------------------------------

def test_parameters_with_bias():
    layer = Linear(4, 8, bias=True)
    params = list(layer.parameters())
    assert len(params) == 2

def test_parameters_no_bias():
    layer = Linear(4, 8, bias=False)
    params = list(layer.parameters())
    assert len(params) == 1

def test_weight_shape():
    layer = Linear(4, 8)
    assert layer.weight.shape == (8, 4)

def test_bias_shape():
    layer = Linear(4, 8)
    assert layer.bias.shape == (8,)


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
