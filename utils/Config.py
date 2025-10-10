from dataclasses import dataclass
import yaml
from pathlib import Path
import tiktoken

current_dir = Path(__file__).parent
config = yaml.safe_load(open(current_dir / "config.yaml", "r"))

@dataclass
class DataConfig:
    batch_size: int = config["data"]["batch_size"]
    train_path: str = config["data"]["train_path"]
    val_path: str = config["data"]["val_path"]

@dataclass
class GPTConfig:
    vocab_size: int = tiktoken.get_encoding(config["gpt"]["encoder_name"]).n_vocab
    embed_dim: int = config["gpt"]["embed_dim"]
    token_len: int = config["gpt"]["token_len"]
    n_head: int = config["gpt"]["n_head"]
    ff_dim: int = config["gpt"]["ff_dim"]
    n_blocks: int = config["gpt"]["n_blocks"]

@dataclass
class TrainerConfig:
    seed: int = 0
    lr: float = 1e-3
    patient: int = 20
    optimizer: str = "Adam"
    

'''
Test Config
'''


@dataclass
class TestDataConfig:
    batch_size: int = config["test"]["data"]["batch_size"]
    train_path: str = config["test"]["data"]["train_path"]
    val_path: str = config["test"]["data"]["val_path"]


@dataclass
class TestGPTConfig:
    vocab_size: int = tiktoken.get_encoding(config["test"]["gpt"]["encoder_name"]).n_vocab
    embed_dim: int = config["test"]["gpt"]["embed_dim"]
    token_len: int = config["test"]["gpt"]["token_len"]
    n_head: int = config["test"]["gpt"]["n_head"]
    ff_dim: int = config["test"]["gpt"]["ff_dim"]
    n_blocks: int = config["test"]["gpt"]["n_blocks"]

@dataclass
class TestTrainerConfig:
    seed: int = 0
    lr: float = 1e-3
    patient: int = 20
    optimizer: str = "Adam"