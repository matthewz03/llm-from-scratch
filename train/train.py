import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import math
import time
from data.TokenDataset import TokenDataset
from modules.LanguageModel import LanguageModel
from modules.CrossEntropyLoss import CrossEntropyLoss
from optim.AdamW import AdamW
from torch.utils.data import DataLoader
import argparse


def load_data(train_dataset_path, val_dataset_path, batch_size, seq_len):
    train_dataset = TokenDataset(train_dataset_path, seq_len)
    val_dataset   = TokenDataset(val_dataset_path,   seq_len)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device, max_batches):
    model.eval()
    total_loss, n = 0.0, 0
    for inputs, targets in val_loader:
        if n >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        total_loss += loss_fn(model(inputs), targets).item()
        n += 1
    model.train()
    return total_loss / n if n > 0 else float('nan')


def main():
    parser = argparse.ArgumentParser(description="Main training entrypoint for LLM")
    parser.add_argument("--train_path",      type=str,   default="../tinystories/data/train/train.bin")
    parser.add_argument("--val_path",        type=str,   default="../tinystories/data/val/val.bin")
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--seq_len",         type=int,   default=256)
    parser.add_argument("--vocab_size",      type=int,   default=50257)
    parser.add_argument("--dim_model",       type=int,   default=256)
    parser.add_argument("--n_blocks",        type=int,   default=4)
    parser.add_argument("--dim_feedforward", type=int,   default=512)
    parser.add_argument("--n_attn_heads",    type=int,   default=8)
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--log_every",       type=int,   default=50,  help="Log train metrics every N steps")
    parser.add_argument("--val_every",       type=int,   default=500, help="Run validation every N steps")
    parser.add_argument("--val_batches",     type=int,   default=50,  help="Number of val batches per evaluation")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = load_data(
        args.train_path, args.val_path, args.batch_size, args.seq_len
    )
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    model = LanguageModel(
        d_model=args.dim_model,
        vocab_size=args.vocab_size,
        n_blocks=args.n_blocks,
        dim_feedforward=args.dim_feedforward,
        n_attn_heads=args.n_attn_heads,
        rope_seq_len=args.seq_len,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn   = CrossEntropyLoss()

    step = 0
    running_loss = 0.0
    tokens_in_window = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            loss = loss_fn(model(inputs), targets)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            optimizer.step()

            running_loss     += loss.item()
            tokens_in_window += args.batch_size * args.seq_len
            step             += 1

            if step % args.log_every == 0:
                elapsed   = time.time() - t0
                avg_loss  = running_loss / args.log_every
                tok_per_s = tokens_in_window / elapsed
                lr        = optimizer.param_groups[0]['lr']
                print(
                    f"step {step:6d} | epoch {epoch} | "
                    f"loss {avg_loss:.4f} | ppl {math.exp(avg_loss):8.2f} | "
                    f"grad_norm {grad_norm:.3f} | "
                    f"lr {lr:.2e} | "
                    f"tok/s {tok_per_s:,.0f}"
                )
                running_loss     = 0.0
                tokens_in_window = 0
                t0 = time.time()

            if step % args.val_every == 0:
                val_loss = evaluate(model, val_loader, loss_fn, device, args.val_batches)
                print(
                    f"step {step:6d} | "
                    f"VAL loss {val_loss:.4f} | VAL ppl {math.exp(val_loss):.2f}"
                )


if __name__ == "__main__":
    main()
