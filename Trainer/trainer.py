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

train_loss_list = {}
val_loss_list = {}

train_loss_acc = []
val_loss_acc = []

train_loss_per_iter_eval = {}
val_loss_per_iter_eval = {}

gpt_model.to(device)

for _idx in range(max_iters):
    print("="*10)
    print(f"Iters: {_idx+1}/{max_iters}")
    x,y = data_loader.get_train_batch()
    x,y = x.to(device), y.to(device)
    pred = gpt_model(x)
    B, T, C = pred.size()
    train_loss = criterion(pred.view(B*T, C), y.view(B*T))
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_loss_acc.append({"batch_size":B, "train_loss_sum":train_loss.item() * B})

    with torch.no_grad():
        x,y = data_loader.get_val_batch()
        x,y = x.to(device), y.to(device)
        gpt_model.to(device)
        pred = gpt_model(x)
        B, T, C = pred.size()
        val_loss = criterion(pred.view(B*T, C), y.view(B*T))
    val_loss_acc.append({"batch_size":B, "val_loss_sum":val_loss.item() * B})

    if _idx % config['deep_learning_settings']['trainer_config']['iter_eval'] == 0:
        sum_of_train_loss = sum(list(map(lambda x: x['train_loss_sum'], train_loss_acc)))
        sum_of_batch_size = sum(list(map(lambda x: x['batch_size'], train_loss_acc)))
        train_loss_per_iter_eval = sum_of_train_loss / sum_of_batch_size
        train_loss_list[_idx] = train_loss_per_iter_eval
        train_loss_acc = []
        print(f"{_idx}th iter train loss:{train_loss_per_iter_eval:.3f}")

        
        val_loss_per_iter_eval = sum(list(map(lambda x: x['val_loss_sum'], val_loss_acc))) / sum(list(map(lambda x: x['batch_size'], val_loss_acc)))
        val_loss_list[_idx] = val_loss_per_iter_eval
        val_loss_acc = []
        print(f"{_idx}th iter val loss:{val_loss_per_iter_eval:.3f}")