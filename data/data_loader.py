import numpy as np
import torch
import random


import sys

sys.path.append("..")
from root_path import get_root_path

proj_dir = str(get_root_path())

train_set = np.memmap(f"{proj_dir}/data/train.bin", dtype=np.uint16, mode='r').astype(np.int64)
val_set = np.memmap(f"{proj_dir}/data/val.bin", dtype=np.uint16, mode='r').astype(np.int64)





class DataLoader:
    def __init__(self, dataset):
        self.token_len = token_len
        self.batch_size = batch_size


    def get_batch(self):
        indice = random.sample(self.sample_pool, k=self.batch_size)
        data = torch.tensor([self.data_set[_idx : _idx+self.token_len+1].tolist() for _idx in indice])
        x = data[:,:-1]
        y = data[:,1:]
        return x, y










class TokenDataLoader:
    def __init__(self, mode:str, token_len:int, batch_size:int):
        assert (mode=='train') or (mode=='val'), "The mode should be either train or val"
        if mode=="train":
            self.data_set = train_set
        elif mode=="val":
            self.data_set = val_set
        self.sample_pool = range(len(train_set) - token_len)

        self.token_len = token_len
        self.batch_size = batch_size


    def get_batch(self):
        indice = random.sample(self.sample_pool, k=self.batch_size)
        data = torch.tensor([self.data_set[_idx : _idx+self.token_len+1].tolist() for _idx in indice])
        x = data[:,:-1]
        y = data[:,1:]
        return x, y