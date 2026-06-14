import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from modules.CrossEntropyLoss import CrossEntropyLoss, CrossEntropyLossFunction

B, C, SEQ = 8, 5, 6


# ---------------------------------------------------------------------------
# Forward — index targets
# ---------------------------------------------------------------------------

def test_forward_index_mean_matches_ref():
    ref = nn.CrossEntropyLoss(reduction='mean')
    ours = CrossEntropyLoss(reduction='mean')
    x = torch.randn(B, C)
    t = torch.randint(0, C, (B,))
    assert torch.allclose(ours(x, t), ref(x, t), atol=1e-5)


def test_forward_index_sum_matches_ref():
    ref = nn.CrossEntropyLoss(reduction='sum')
    ours = CrossEntropyLoss(reduction='sum')
    x = torch.randn(B, C)
    t = torch.randint(0, C, (B,))
    assert torch.allclose(ours(x, t), ref(x, t), atol=1e-5)


def test_forward_index_3d_matches_ref():
    """Batch x seq x class input with per-token targets."""
    ref = nn.CrossEntropyLoss(reduction='mean')
    ours = CrossEntropyLoss(reduction='mean')
    x = torch.randn(B, SEQ, C)
    t = torch.randint(0, C, (B, SEQ))
    assert torch.allclose(ours(x, t), ref(x.reshape(-1, C), t.reshape(-1)), atol=1e-5)


def test_forward_output_is_scalar():
    ours = CrossEntropyLoss()
    x = torch.randn(B, C)
    t = torch.randint(0, C, (B,))
    assert ours(x, t).shape == ()


def test_forward_loss_is_positive():
    ours = CrossEntropyLoss()
    x = torch.randn(B, C)
    t = torch.randint(0, C, (B,))
    assert ours(x, t).item() > 0


def test_forward_perfect_prediction_low_loss():
    """Very confident correct predictions should give near-zero loss."""
    ours = CrossEntropyLoss()
    x = torch.zeros(4, C)
    x[:, 0] = 100.0  # confident on class 0
    t = torch.zeros(4, dtype=torch.long)  # all target class 0
    assert ours(x, t).item() < 1e-3


def test_forward_wrong_prediction_high_loss():
    """Confident wrong prediction should give high loss."""
    ours = CrossEntropyLoss()
    x = torch.zeros(4, C)
    x[:, 0] = 100.0   # confident on class 0
    t = torch.ones(4, dtype=torch.long)  # target is class 1
    assert ours(x, t).item() > 50.0


# ---------------------------------------------------------------------------
# Forward — soft targets (sum reduction matches PyTorch; mean does not since
# PyTorch sums over classes before averaging over batch, but ours uses .mean()
# over all elements — so only sum is tested against reference)
# ---------------------------------------------------------------------------

def test_forward_soft_sum_matches_ref():
    ref = nn.CrossEntropyLoss(reduction='sum')
    ours = CrossEntropyLoss(reduction='sum')
    x = torch.randn(B, C)
    t = torch.softmax(torch.randn(B, C), dim=-1)
    assert torch.allclose(ours(x, t), ref(x, t), atol=1e-5)


def test_forward_soft_mean_matches_ref():
    ref = nn.CrossEntropyLoss(reduction='mean')
    ours = CrossEntropyLoss(reduction='mean')
    x = torch.randn(B, C)
    t = torch.softmax(torch.randn(B, C), dim=-1)
    assert torch.allclose(ours(x, t), ref(x, t), atol=1e-5)


def test_grad_soft_mean_matches_ref():
    ref = nn.CrossEntropyLoss(reduction='mean')
    ours = CrossEntropyLoss(reduction='mean')
    x1 = torch.randn(B, C, requires_grad=True)
    x2 = x1.detach().requires_grad_(True)
    t = torch.softmax(torch.randn(B, C), dim=-1)
    ours(x1, t).backward()
    ref(x2, t).backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5), \
        f"grad mismatch, max diff: {(x1.grad - x2.grad).abs().max()}"


def test_forward_soft_uniform_target():
    """Uniform target should give loss = log(C) regardless of logits."""
    ours = CrossEntropyLoss(reduction='mean')
    x = torch.randn(B, C)
    t = torch.full((B, C), 1.0 / C)
    loss = ours(x, t).item()
    # uniform CE = -sum(1/C * log(p)) = -mean(log_p) weighted uniformly
    # This is a sanity check that loss is positive and finite
    assert loss > 0 and torch.isfinite(torch.tensor(loss))


# ---------------------------------------------------------------------------
# Backward — index targets
# ---------------------------------------------------------------------------

def test_grad_index_matches_ref():
    ref = nn.CrossEntropyLoss(reduction='mean')
    ours = CrossEntropyLoss(reduction='mean')
    x1 = torch.randn(B, C, requires_grad=True)
    x2 = x1.detach().requires_grad_(True)
    t = torch.randint(0, C, (B,))
    ours(x1, t).backward()
    ref(x2, t).backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5), \
        f"grad mismatch, max diff: {(x1.grad - x2.grad).abs().max()}"


def test_grad_index_sum_matches_ref():
    ref = nn.CrossEntropyLoss(reduction='sum')
    ours = CrossEntropyLoss(reduction='sum')
    x1 = torch.randn(B, C, requires_grad=True)
    x2 = x1.detach().requires_grad_(True)
    t = torch.randint(0, C, (B,))
    ours(x1, t).backward()
    ref(x2, t).backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)


def test_gradcheck_index():
    x = torch.randn(4, C, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, C, (4,))
    assert torch.autograd.gradcheck(
        lambda x: CrossEntropyLossFunction.apply(x, t, 'mean'),
        (x,), eps=1e-6, atol=1e-4
    )


def test_gradcheck_index_sum():
    x = torch.randn(4, C, dtype=torch.float64, requires_grad=True)
    t = torch.randint(0, C, (4,))
    assert torch.autograd.gradcheck(
        lambda x: CrossEntropyLossFunction.apply(x, t, 'sum'),
        (x,), eps=1e-6, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Backward — soft targets
# ---------------------------------------------------------------------------

def test_gradcheck_soft_mean():
    x = torch.randn(4, C, dtype=torch.float64, requires_grad=True)
    t = torch.softmax(torch.randn(4, C, dtype=torch.float64), dim=-1)
    assert torch.autograd.gradcheck(
        lambda x: CrossEntropyLossFunction.apply(x, t, 'mean'),
        (x,), eps=1e-6, atol=1e-4
    )


def test_gradcheck_soft_sum():
    x = torch.randn(4, C, dtype=torch.float64, requires_grad=True)
    t = torch.softmax(torch.randn(4, C, dtype=torch.float64), dim=-1)
    assert torch.autograd.gradcheck(
        lambda x: CrossEntropyLossFunction.apply(x, t, 'sum'),
        (x,), eps=1e-6, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Module correctness
# ---------------------------------------------------------------------------

def test_no_learnable_parameters():
    assert len(list(CrossEntropyLoss().parameters())) == 0


def test_invalid_reduction_raises():
    try:
        CrossEntropyLoss(reduction='none')
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
