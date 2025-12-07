import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tiktoken
import numpy as np

from beartype import beartype

import sys

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size:int , embed_dim:int, token_len:int, dropout:float):
        super().__init__()
        self.input_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pe = nn.Embedding(num_embeddings=token_len, embedding_dim=embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, token_ids) - contains token ids
        batch_size, token_ids = x.shape
        
        
        # Create position indices: (token_ids,) containing [0, 1, 2, ..., token_ids-1]
        assert token_ids <= self.pe.num_embeddings, \
            f"seq_len={token_ids} exceeds max token_len={self.pe.num_embeddings}"
        positions = torch.arange(token_ids, dtype=torch.long, device=x.device)
        
        # Get token embeddings: (batch_size, token_ids, embed_dim)
        token_embeddings = self.input_embedding(x)
        
        # Get positional embeddings: (token_ids, embed_dim)
        # Broadcasting will handle adding this to each batch
        positional_embeddings = self.pe(positions)
        
        # Add them together: (batch_size, token_ids, embed_dim)
        result = self.dropout(token_embeddings + positional_embeddings)
        
        return result

class MHA(nn.Module):
    def __init__(self, embed_dim:int, n_head:int, dropout:float):
        super().__init__()
        assert embed_dim % n_head == 0, \
            f"embed_dim ({embed_dim}) must be divisible by n_head ({n_head})"        
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.QKV_embed = nn.Linear(self.embed_dim, self.embed_dim*3)
        self.linear_o = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout_ratio = dropout
        self.dropout = nn.Dropout(dropout)

        """
        2025.11.11 기준:
        Cursor Tab의 단점 발견
        - scaled_dot_product_attention 이 맞는데, Scaled_Dot_Product_Attention를 Auto complete 한다.
        """
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate Q, K, V
        '''
        Diffrent with nanoGPT, -> the order of process
        Mine:
            - split multi-heads and then split q, k, v
        nanoGPT
            - split q, k, v first. Then split multi-heads
        '''
        qkv = self.QKV_embed(x)
        qkv = qkv.view(B, T, self.n_head, self.embed_dim*3//self.n_head)
        q, k, v = qkv.split(self.embed_dim//self.n_head, dim=-1)

        q = q.transpose(1,2) # (batch, n_head, token_length, embedding_dim//n_head)
        k = k.transpose(1,2) # (batch, n_head, token_length, embedding_dim//n_head)
        v = v.transpose(1,2) # (batch, n_head, token_length, embedding_dim//n_head)

        '''Start conditional branch of sdpa and causal attention'''
        result_per_MH = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_ratio, is_causal=True)

        '''
        END conditional branch of sdpa and causal attention
        The operation of SDPA is end at attention @ v
        '''

        result = result_per_MH.transpose(1,2).contiguous().view(B, T, C)

        pro_o = self.dropout(self.linear_o(result))

        return pro_o

class FFN(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.gelu = nn.GELU()
        self.Linear_in = nn.Linear(embed_dim, embed_dim*4)
        self.Linear_out = nn.Linear(embed_dim*4, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.Linear_out(self.gelu(self.Linear_in(x))))

# MHA + FNN
class Block(nn.Module):
    def __init__(self, embed_dim, n_head, dropout:float):
        super().__init__()
        self.ln_MHA = nn.LayerNorm(embed_dim)
        self.ln_FFN = nn.LayerNorm(embed_dim)
        self.MHA = MHA(embed_dim=embed_dim, n_head=n_head, dropout=dropout)
        self.FFN = FFN(embed_dim=embed_dim, dropout=dropout)

    def forward(self, x):
        x = self.MHA(self.ln_MHA(x))+x
        x = self.FFN(self.ln_FFN(x))+x
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, token_len:int, n_head:int, n_blocks:int, dropout:float):
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, token_len=token_len, dropout=dropout)
        self.blocks = nn.ModuleList([Block(embed_dim=embed_dim, n_head=n_head, dropout=dropout) for _ in range(n_blocks)])
        self.linear_out = nn.Linear(embed_dim, vocab_size)
#        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.input_embedding(x)
        for _block in self.blocks:
            x = _block(x)
        x = self.linear_out(x)
#        x = self.softmax(x)
        """
        softmax는 CrossEntropyLoss에 포함되어 있으므로 모델에서 softmax를 적용하지 않는다.
        """
        return x