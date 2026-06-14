import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn as nn
from modules.Embedding import Embedding, EmbeddingFunction

V = 16   # vocab size
D = 8    # embedding dim
BATCH = 4
SEQ = 6


def make_ref(our_layer):
    ref = nn.Embedding(our_layer.embeddings.shape[0], our_layer.embeddings.shape[1])
    with torch.no_grad():
        ref.weight.copy_(our_layer.embeddings)
    return ref


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def test_output_shape():
    layer = Embedding(V, D)
    idx = torch.randint(0, V, (BATCH, SEQ))
    assert layer(idx).shape == (BATCH, SEQ, D)


def test_output_shape_1d():
    layer = Embedding(V, D)
    idx = torch.randint(0, V, (SEQ,))
    assert layer(idx).shape == (SEQ, D)


def test_matches_ref():
    layer = Embedding(V, D)
    ref = make_ref(layer)
    idx = torch.randint(0, V, (BATCH, SEQ))
    assert torch.allclose(layer(idx).float(), ref(idx))


def test_lookup_correct_row():
    layer = Embedding(V, D)
    idx = torch.tensor([3])
    out = layer(idx)
    assert torch.allclose(out[0], layer.embeddings[3])


def test_same_index_same_output():
    layer = Embedding(V, D)
    idx = torch.tensor([2, 5, 2])
    out = layer(idx)
    assert torch.allclose(out[0], out[2])


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

def test_grad_accumulates_for_repeated_index():
    """When the same index appears twice, its embedding grad should be the sum."""
    layer = Embedding(V, D)
    ref = make_ref(layer)

    idx = torch.tensor([1, 1])
    x1 = layer(idx)
    x1.sum().backward()

    x2 = ref(idx)
    x2.sum().backward()

    assert torch.allclose(layer.embeddings.grad[1], ref.weight.grad[1], atol=1e-6), \
        "grad for repeated index should accumulate"


def test_grad_zero_for_unvisited_index():
    """Indices not in the batch should have zero gradient."""
    layer = Embedding(V, D)
    idx = torch.tensor([0, 1, 2])
    layer(idx).sum().backward()
    assert torch.all(layer.embeddings.grad[5:] == 0), \
        "unvisited embeddings should have zero gradient"


def test_grad_match_ref():
    layer = Embedding(V, D)
    ref = make_ref(layer)

    idx = torch.randint(0, V, (BATCH, SEQ))
    layer(idx).sum().backward()
    ref(idx).sum().backward()

    assert torch.allclose(layer.embeddings.grad, ref.weight.grad, atol=1e-6), \
        f"grad mismatch, max diff: {(layer.embeddings.grad - ref.weight.grad).abs().max()}"


def test_gradcheck():
    E = EmbeddingFunction
    emb = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    idx = torch.randint(0, 8, (2, 3))
    assert torch.autograd.gradcheck(lambda e: E.apply(e, idx), (emb,), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# Module correctness
# ---------------------------------------------------------------------------

def test_parameter_count():
    assert len(list(Embedding(V, D).parameters())) == 1


def test_embedding_shape():
    layer = Embedding(V, D)
    assert layer.embeddings.shape == (V, D)


def test_init_is_normal():
    torch.manual_seed(0)
    layer = Embedding(1000, 64)
    mean = layer.embeddings.data.mean().item()
    std = layer.embeddings.data.std().item()
    assert abs(mean) < 0.1, f"mean far from 0: {mean}"
    assert abs(std - 1.0) < 0.1, f"std far from 1: {std}"


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
