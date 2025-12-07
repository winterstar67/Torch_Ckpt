import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tiktoken

import numpy as np

from beartype import beartype
import sys
sys.path.append("..")

from utils import torch_ckpt
from model import model

import json
with open("../config.json", "r") as f:
    config = json.load(f)

from root_path import get_root_path
proj_dir = get_root_path()



file_path = str(proj_dir/"Trainer"/"Trainer.ipynb")

config['path_settings']['file_path'] = file_path
config['git_settings']['strict_git'] = False

device = config['env_settings']['device']
epoch = config['deep_learning_settings']['trainer_config']['max_epochs']


seed_test = torch_ckpt.ckpt_manager(**config)
print(torch.rand([2,9]))

seed_test = torch_ckpt.ckpt_manager(**config)
print(torch.rand([2,9]))

seed_test = torch_ckpt.ckpt_manager(**config)
print(torch.rand([2,9]))

print("Repeatly call it to check if the seed is set correctly or not")

from data.data_loader import TokenDataLoader

data_loader = TokenDataLoader(mode='train', 
                         token_len=config['deep_learning_settings']['model_config']['token_len'], 
                         batch_size=config['deep_learning_settings']['data_config']['batch_size'])


gpt_model = model.GPT(**config['deep_learning_settings']['model_config'])

criterion = nn.CrossEntropyLoss()  # 예: 분류 문제일 때
optimizer = optim.Adam(gpt_model.parameters(), lr=config['deep_learning_settings']['optimizer_config']['lr'])


from root_path import get_root_path
import random
import torch
import numpy as np

print("="*50)
print("trainer start:")
print("device:", device)
print("="*50)
for _idx in range(10):
    print("="*10)
    print(f"Epoch {_idx+1}/{epoch}")
    # x,y = data_loader.get_batch()
    x,y = data_loader.get_batch() # Test
    x,y = x.to(device), y.to(device)
    gpt_model.to(device)
    pred = gpt_model(x)
    B, T, C = pred.size()
    loss = criterion(pred.view(B*T, C), y.view(B*T))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"loss:{loss:.3f}")