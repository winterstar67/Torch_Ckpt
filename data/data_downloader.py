"""
Ordering: First(1)

Data preparation referring to nanoGPT/data/shakespeare/prepare.py  
- It's not simple char token based data (word based token data is more realistic)
- Not too heavy (enough data for testing)
"""

import os
import requests
import tiktoken
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
f"""
GPT 답변:
    dtype=np.uint16인 이유는, GPT-2 BPE의 토큰 수가 약 50,000개 정도이기 때문이야.
    uint16은 0–65,535 범위를 표현하므로 충분하지.

np.save를 통한 npy 확장자보다 bin이 좋은 이유:
- 핵심 포인트: 메모리 매핑(np.memmap)과 스트리밍 학습
    Karpathy는 이후 데이터 로딩에서 이렇게 함:
    data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    “스트리밍(memmap)”은 대규모 데이터를 전부 메모리에 올리지 않고, 디스크에 둔 채로 필요한 부분만 즉시 접근하는 방식을 말해.

np.memmap은 이렇게 작동해:
    “파일 전체를 메모리에 올리지 말고,
    필요한 조각(chunk)만 디스크에서 그때그때 읽자.”

"""

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
