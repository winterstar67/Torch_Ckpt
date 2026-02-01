import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tiktoken

import numpy as np

from beartype import beartype
import json
import glob

def is_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

project_name = "nanoGPT"  # variable depends on project

if is_colab():
    from google.colab import drive
    drive.mount('COLAB_DRIVE')

    matches = glob.glob(f"COLAB_DRIVE/MyDrive/**/{project_name}/root_path.py", recursive=True)
    print(matches)
    if len(matches) == 1:
        proj_dir = os.path.dirname(matches[0])
    with open(os.path.join(proj_dir, "config.json"), "r") as f:
        config = json.load(f)
    config['seed_settings']['use_seed'] = False

else:
    from root_path import get_root_path
    proj_dir = get_root_path()
    print("Is local")
    with open(os.path.join(proj_dir, "config.json"), "r") as f:
        config = json.load(f)

import sys
sys.path.append(proj_dir)

from utils import torch_ckpt
from model import model

file_path = os.path.join(proj_dir, "Trainer", "Trainer.ipynb")

config['path_settings']['file_path'] = file_path
config['git_settings']['strict_git'] = False

device = config['env_settings']['device']
max_iters = config['deep_learning_settings']['trainer_config']['max_iters']

# Seed Setting
seed_test = torch_ckpt.ckpt_manager(**config)

# Data Preparation
from data.data_loader import TokenDataLoader

data_loader = TokenDataLoader(data_train_path=os.path.join(proj_dir, "data", "tokenized", "concatenated", "train_all.bin"),
                              data_val_path=os.path.join(proj_dir, "data", "tokenized", "concatenated", "val_all_10_to_11.bin"),
                              token_len=config['deep_learning_settings']['model_config']['token_len'],
                              batch_size=config['deep_learning_settings']['data_config']['batch_size'])

# Model definition
if "ff_dim" in config['deep_learning_settings']['model_config'].keys():
    config['deep_learning_settings']['model_config'].pop('ff_dim', None)

gpt_model = model.GPT(**config['deep_learning_settings']['model_config'])

# Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gpt_model.parameters(), lr=config['deep_learning_settings']['optimizer_config']['lr'])

# Training Loop
import random

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
