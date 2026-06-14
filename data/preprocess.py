import tiktoken
import numpy as np
import pyarrow.dataset as ds
import argparse
from pathlib import Path
import os
import multiprocessing as mp

def _shard_pattern(output_path: Path) -> str:
    return output_path.stem + "_*.bin"

def _existing_shards(output_path: Path):
    return list(output_path.parent.glob(_shard_pattern(output_path)))

def _cleanup(output_path: Path):
    for shard in _existing_shards(output_path):
        shard.unlink()

def _concat_shards(output_path: Path):
    with open(output_path, "wb") as f:
        for shard in sorted(_existing_shards(output_path)):
            np.fromfile(shard, dtype=np.uint16).tofile(f)
            shard.unlink()

def _worker(text: str, output_path: str):
    tokenizer = tiktoken.get_encoding("gpt2")
    output_path = Path(output_path)
    shard_path = output_path.with_stem(output_path.stem + "_" + str(os.getpid()))
    with open(shard_path, "ab") as f:
        tokens = tokenizer.encode_ordinary(text) + [tokenizer.eot_token]
        np.array(tokens, dtype=np.uint16).tofile(f)

def preprocess(train_path: str, val_path: str, output_train_path: str, output_val_path: str, batch_size: int = 1_000_000):
    train_dataset = ds.dataset(train_path, format="parquet")
    val_dataset = ds.dataset(val_path, format="parquet")

    for dataset, output_path in [(train_dataset, output_train_path), (val_dataset, output_val_path)]:
        output_path = Path(output_path)
        _cleanup(output_path)
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for batch in dataset.to_batches(batch_size=batch_size):
                pool.starmap(_worker, [(t, str(output_path)) for t in batch["text"].tolist()])
        _concat_shards(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text data into token ids and save as binary files.")
    parser.add_argument("--train_path", type=str, default="../tinystories/data/train", help="Path to the training data (parquet file).")
    parser.add_argument("--val_path", type=str, default="../tinystories/data/val", help="Path to the validation data (parquet file).")
    parser.add_argument("--output_train_path", type=str, default="../tinystories/data/train/train.bin", help="Path to save the preprocessed training data (binary file).")
    parser.add_argument("--output_val_path", type=str, default="../tinystories/data/val/val.bin", help="Path to save the preprocessed validation data (binary file).")
    parser.add_argument("--batch_size", type=int, default=1_000_000, help="Batch size for processing the data.")
    

    args = parser.parse_args()

    preprocess(args.train_path, args.val_path, args.output_train_path, args.output_val_path, args.batch_size)



