import torch
from torch.utils.data import Dataset
import numpy as np

class TokenDataset(Dataset):

    def __init__(self, path: str, seq_len: int):
        super().__init__()

        self.tokens = np.memmap(path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.num_windows = len(self.tokens) - seq_len

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        chunk = self.tokens[index : index + self.seq_len + 1].astype(np.int64)
        tensor_chunk = torch.from_numpy(chunk)
        return tensor_chunk[:-1], tensor_chunk[1:]
