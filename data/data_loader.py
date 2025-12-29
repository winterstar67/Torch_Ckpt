import numpy as np
import torch
import random

class TokenDataLoader:
    def __init__(self, data_train_path:str, data_val_path:str, token_len:int, batch_size:int, seed:int=42):

        self.data_train_set = np.memmap(f"{data_train_path}", dtype=np.uint16, mode='r')
        self.data_val_set = np.memmap(f"{data_val_path}", dtype=np.uint16, mode='r')

        self.token_len = token_len
        self.batch_size = batch_size
        self.train_max_start = len(self.data_train_set) - token_len
        self.val_max_start = len(self.data_val_set) - token_len

        self.rng = np.random.default_rng(seed)

    def get_train_batch(self):
        start_idx = self.rng.choice(self.train_max_start, size=self.batch_size, replace=False)
        
        offset = np.arange(self.token_len)
        batch_u16 = self.data_train_set[start_idx[:, None] + offset[None, :]]

        # convert ONLY this batch to int64 for embedding indexing
        # batch = torch.from_numpy(batch_u16.astype(np.int64, copy=False)) -> this makes memmap wasteful.
        batch = torch.from_numpy(batch_u16).to(torch.int64)
        x = batch[:,:-1]
        y = batch[:,1:]
        return x, y

    def get_val_batch(self):
        start_idx = self.rng.choice(self.val_max_start, size=self.batch_size, replace=False)
        
        offset = np.arange(self.token_len)
        # Can remove this comment line
        batch_u16 = self.data_val_set[start_idx[:, None] + offset[None, :]]

        # convert ONLY this batch to int64 for embedding indexing
        # batch = torch.from_numpy(batch_u16.astype(np.int64, copy=False)) -> this makes memmap wasteful.
        batch = torch.from_numpy(batch_u16).to(torch.int64)
        x = batch[:,:-1]
        y = batch[:,1:]
        return x, y