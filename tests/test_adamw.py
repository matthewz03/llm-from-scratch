import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn as nn
from optim.AdamW import AdamW


def make_pair(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Two identical linear layers with our AdamW and PyTorch's AdamW."""
    torch.manual_seed(0)
    ours = nn.Linear(8, 4)
    ref = nn.Linear(8, 4)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
        ref.bias.copy_(ours.bias)
    opt_ours = AdamW(ours.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return ours, ref, opt_ours, opt_ref


def step_both(ours, ref, opt_ours, opt_ref, x):
    """Run one forward+backward+step on both models with the same input."""
    loss_ours = ours(x).sum()
    loss_ref = ref(x).sum()
    opt_ours.zero_grad(); loss_ours.backward(); opt_ours.step()
    opt_ref.zero_grad();  loss_ref.backward();  opt_ref.step()


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------

def test_params_change_after_step():
    ours, _, opt_ours, _ = make_pair()
    w_before = ours.weight.data.clone()
    x = torch.randn(4, 8)
    opt_ours.zero_grad()
    ours(x).sum().backward()
    opt_ours.step()
    assert not torch.allclose(ours.weight.data, w_before), "weights should change after a step"


def test_no_grad_param_unchanged():
    """Parameters with None grad must not be updated."""
    ours, _, opt_ours, _ = make_pair()
    w_before = ours.weight.data.clone()
    # step without calling backward — grads stay None
    opt_ours.step()
    assert torch.allclose(ours.weight.data, w_before), "param with None grad should be unchanged"


# ---------------------------------------------------------------------------
# Matches PyTorch AdamW
# ---------------------------------------------------------------------------

def _match_ref(n_steps, lr, betas, eps, weight_decay):
    ours, ref, opt_ours, opt_ref = make_pair(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    torch.manual_seed(1)
    for _ in range(n_steps):
        x = torch.randn(4, 8)
        step_both(ours, ref, opt_ours, opt_ref, x)
    assert torch.allclose(ours.weight.data, ref.weight.data, atol=1e-6), \
        f"weight mismatch after {n_steps} steps, max diff: {(ours.weight.data - ref.weight.data).abs().max()}"
    assert torch.allclose(ours.bias.data, ref.bias.data, atol=1e-6), \
        f"bias mismatch after {n_steps} steps"


def test_matches_ref_one_step():
    _match_ref(1, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

def test_matches_ref_ten_steps():
    _match_ref(10, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

def test_matches_ref_with_weight_decay():
    _match_ref(10, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

def test_matches_ref_different_betas():
    _match_ref(10, lr=5e-4, betas=(0.95, 0.98), eps=1e-8, weight_decay=0.01)


# ---------------------------------------------------------------------------
# Weight decay behaviour
# ---------------------------------------------------------------------------

def test_weight_decay_shrinks_params():
    """With zero gradient and weight_decay > 0, parameters should shrink each step."""
    torch.manual_seed(0)
    layer = nn.Linear(4, 2, bias=False)
    opt = AdamW(layer.parameters(), lr=1e-2, weight_decay=1.0)

    # inject a constant gradient of zero
    layer.weight.grad = torch.zeros_like(layer.weight)
    norms = []
    for _ in range(20):
        opt.step()
        norms.append(layer.weight.data.norm().item())
        layer.weight.grad = torch.zeros_like(layer.weight)

    assert norms[-1] < norms[0], "weight decay should reduce parameter norm over time"


def test_zero_weight_decay_matches_adam():
    """weight_decay=0 should be identical to plain Adam."""
    _match_ref(10, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


# ---------------------------------------------------------------------------
# Moment state
# ---------------------------------------------------------------------------

def test_moment_state_initialized_on_first_step():
    torch.manual_seed(0)
    layer = nn.Linear(4, 2, bias=False)
    opt = AdamW(layer.parameters(), lr=1e-3)
    layer(torch.randn(3, 4)).sum().backward()
    opt.step()
    state = opt.state[layer.weight]
    assert 'm' in state and 'v' in state, "state should contain m and v after first step"


def test_moment_values_after_one_step():
    """m and v after step 1 should match the closed-form formula."""
    torch.manual_seed(0)
    layer = nn.Linear(4, 2, bias=False)
    beta1, beta2 = 0.9, 0.999
    opt = AdamW(layer.parameters(), lr=1e-3, betas=(beta1, beta2), weight_decay=0.0)

    x = torch.randn(3, 4)
    layer(x).sum().backward()
    grad = layer.weight.grad.clone()
    opt.step()

    state = opt.state[layer.weight]
    expected_m = (1 - beta1) * grad
    expected_v = (1 - beta2) * grad ** 2
    assert torch.allclose(state['m'], expected_m, atol=1e-7), "m after step 1 is wrong"
    assert torch.allclose(state['v'], expected_v, atol=1e-7), "v after step 1 is wrong"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_negative_lr_raises():
    try:
        AdamW(nn.Linear(2, 2).parameters(), lr=-1e-3)
        assert False, "should have raised"
    except ValueError:
        pass

def test_invalid_beta_raises():
    try:
        AdamW(nn.Linear(2, 2).parameters(), betas=(1.5, 0.999))
        assert False, "should have raised"
    except ValueError:
        pass


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
