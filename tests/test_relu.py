import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from modules.ReLU import ReLU, ReLUFunction


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_forward_positive_values():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(ReLU()(x), x)

def test_forward_negative_values():
    x = torch.tensor([-1.0, -2.0, -3.0])
    assert torch.allclose(ReLU()(x), torch.zeros(3))

def test_forward_mixed():
    x = torch.tensor([-1.0, 0.0, 1.0])
    expected = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(ReLU()(x), expected)

def test_forward_matches_ref():
    ref = torch.nn.ReLU()
    layer = ReLU()
    x = torch.randn(16, 8)
    assert torch.allclose(layer(x), ref(x))

def test_forward_3d_matches_ref():
    ref = torch.nn.ReLU()
    layer = ReLU()
    x = torch.randn(4, 10, 8)
    assert torch.allclose(layer(x), ref(x))

# ---------------------------------------------------------------------------
# Inplace
# ---------------------------------------------------------------------------

def test_inplace_output_matches_non_inplace():
    x1 = torch.randn(16, 8)
    x2 = x1.clone()
    out_normal = ReLU(inplace=False)(x1)
    ReLU(inplace=True)(x2)
    assert torch.allclose(out_normal, x2)

def test_inplace_modifies_input():
    x = torch.tensor([-1.0, 0.0, 1.0])
    ReLU(inplace=True)(x)
    assert torch.allclose(x, torch.tensor([0.0, 0.0, 1.0]))

# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def test_grad_matches_ref():
    ref = torch.nn.ReLU()
    layer = ReLU()
    x1 = torch.randn(16, 8, requires_grad=True)
    x2 = x1.detach().requires_grad_(True)
    layer(x1).sum().backward()
    ref(x2).sum().backward()
    assert torch.allclose(x1.grad, x2.grad)

def test_grad_zero_for_negative_inputs():
    x = torch.tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True)
    ReLU()(x).sum().backward()
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert torch.allclose(x.grad, expected)

def test_gradcheck():
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(ReLUFunction.apply, (x, False), eps=1e-6, atol=1e-4)


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
