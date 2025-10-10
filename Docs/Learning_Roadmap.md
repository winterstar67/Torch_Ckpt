# nanoGPT 학습 로드맵

## 학습 목표
**"학습이 전혀 안된 모델에 비해 어느정도 attention을 학습해서 Natural Language Processing이 가능해졌구나"를 파악하기**
   - **자연어 기반의 test task를 준비하자** 
      - **학습이 next token prediction이니까, 이것을 기반으로 결과를 확인하자.**
**"이것을 기반으로 추후 나오는 모델 학습 기법을 테스트해보고 이러저러한 아이디어 검증을 할 예정임"**

---

## Task 1: 목표 설정 및 이해

### 참고할 파일
- **`nanoGPT/README.md`**: 전체 프로젝트 개요 및 Quick Start
- **`nanoGPT/model.py`**: GPT 모델 아키텍처 이해
  - `CausalSelfAttention` 클래스 (line 29-76): Attention 메커니즘의 핵심
  - `Block` 클래스 (line 94-106): Transformer 블록 구조
  - `GPT` 클래스 (line 118-330): 전체 모델 구조

### 수행할 작업
1. **Attention 메커니즘 이해하기**
   - `model.py:29-76` CausalSelfAttention 클래스 분석
   - Query, Key, Value 계산 방식 이해
   - Causal masking이 어떻게 미래 토큰을 가리는지 확인

2. **학습 전후 비교 계획 세우기**
   - 랜덤 초기화된 모델 vs 학습된 모델
   - 같은 프롬프트에 대한 생성 결과 비교
   - Validation Loss 변화 추적

---

## Task 2: 데이터셋 이해 및 준비

### 추천 데이터셋: Shakespeare Character-level
**이유:**
- 작고 빠름 (~1MB, 약 111만 문자)
- GPU 없이도 몇 분 내 학습 가능
- 문자 수준이라 토크나이저 단순함 (65개 문자만)
- Attention 학습 효과를 명확히 관찰 가능

### 참고할 파일
- **`nanoGPT/data/shakespeare_char/readme.md`**: 데이터셋 개요
- **`nanoGPT/data/shakespeare_char/prepare.py`**: 데이터 전처리 스크립트 (문자 수준)
- **`nanoGPT/data/shakespeare/prepare.py`**: GPT-2 BPE 토크나이저 사용 예시
- **`nanoGPT/data/openwebtext/prepare.py`**: 대규모 데이터셋 토크나이징 예시

---

## nanoGPT의 Tokenizer 처리 방식

nanoGPT는 **두 가지 토크나이저 방식**을 지원합니다:

### 방식 1: Character-level Tokenizer (커스텀)
**사용 데이터셋:** `shakespeare_char`

**장점:**
- 매우 단순하고 이해하기 쉬움
- 학습 목적으로 최적
- Vocab 크기가 작음 (65개)

**구현 방법:** (`data/shakespeare_char/prepare.py`)
```python
# 1. 고유 문자 추출
chars = sorted(list(set(data)))  # 텍스트의 모든 고유 문자
vocab_size = len(chars)          # 65개

# 2. 문자 <-> 정수 매핑 생성
stoi = {ch:i for i,ch in enumerate(chars)}  # string to int
itos = {i:ch for i,ch in enumerate(chars)}  # int to string

# 3. Encode/Decode 함수 정의
def encode(s):
    return [stoi[c] for c in s]  # 문자열 → 정수 리스트

def decode(l):
    return ''.join([itos[i] for i in l])  # 정수 리스트 → 문자열

# 4. 데이터 인코딩
train_ids = encode(train_data)
val_ids = encode(val_data)

# 5. 바이너리 파일로 저장
train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile('train.bin')

# 6. 메타 정보 저장 (나중에 디코딩할 때 사용)
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
pickle.dump(meta, open('meta.pkl', 'wb'))
```

**생성 파일:**
- `train.bin`: 인코딩된 학습 데이터 (1,003,854 토큰)
- `val.bin`: 인코딩된 검증 데이터 (111,540 토큰)
- `meta.pkl`: vocab 정보 (인코더/디코더 포함)

---

### 방식 2: GPT-2 BPE Tokenizer (tiktoken)
**사용 데이터셋:** `shakespeare`, `openwebtext`

**장점:**
- 사전 학습된 GPT-2 모델과 호환
- 서브워드 단위 토크나이징 (더 효율적)
- Vocab 크기: 50,257개

**구현 방법:** (`data/shakespeare/prepare.py`)
```python
import tiktoken

# 1. GPT-2 BPE 인코더 로드
enc = tiktoken.get_encoding("gpt2")

# 2. 데이터 인코딩
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

# 3. 바이너리 파일로 저장 (동일)
train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile('train.bin')
```

**BPE (Byte Pair Encoding) 특징:**
- 자주 나오는 문자 조합을 하나의 토큰으로 처리
- 예: "learning" → ["learn", "ing"] 또는 ["lear", "ning"]
- 미등록 단어도 서브워드로 분해하여 처리 가능

---

### 방식 3: 대규모 데이터셋 처리 (openwebtext)
**추가 기능:** 병렬 처리 + Memory-mapped 파일

**구현 방법:** (`data/openwebtext/prepare.py`)
```python
from datasets import load_dataset
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# 1. HuggingFace 데이터셋 로드
dataset = load_dataset("openwebtext", num_proc=8)

# 2. 토크나이징 함수 정의
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)  # End of Text 토큰 추가 (50256)
    return {'ids': ids, 'len': len(ids)}

# 3. 병렬 처리로 전체 데이터셋 토크나이징
tokenized = dataset.map(
    process,
    remove_columns=['text'],
    num_proc=8,  # 8개 프로세스 병렬 처리
)

# 4. Memory-mapped 파일로 저장 (대용량 데이터 효율적 처리)
arr = np.memmap('train.bin', dtype=np.uint16, mode='w+', shape=(arr_len,))
# ... 배치 단위로 쓰기
```

**Memory-mapped 파일의 장점:**
- RAM에 전체 데이터를 로드하지 않음
- 필요한 부분만 디스크에서 읽어옴
- 수십 GB 데이터도 처리 가능

---

### Tokenizer 비교표

| 항목 | Character-level | GPT-2 BPE |
|------|----------------|-----------|
| **Vocab 크기** | ~65개 | 50,257개 |
| **토큰 단위** | 문자 | 서브워드 |
| **구현 난이도** | 매우 쉬움 | 쉬움 (tiktoken 사용) |
| **학습 속도** | 빠름 | 중간 |
| **시퀀스 길이** | 김 (문자 많음) | 짧음 (서브워드 압축) |
| **사전학습 모델** | 호환 안됨 | GPT-2 호환 |
| **사용 예시** | `shakespeare_char` | `shakespeare`, `openwebtext` |

---

### 데이터셋 구조 이해
```python
# prepare.py 주요 부분 (Character-level 기준):
# 1. 데이터 다운로드 (line 13-17)
# 2. 문자 집합 추출 (line 24-27): 65개 고유 문자
# 3. 문자->정수 매핑 생성 (line 30-35)
# 4. Train/Val 분할 (line 38-46): 90% train, 10% val
# 5. 바이너리 파일 저장 (line 48-52): train.bin, val.bin
# 6. 메타 정보 저장 (line 55-61): meta.pkl (인코더/디코더)
```

### 수행할 작업

#### 1. Character-level 데이터 준비 (추천)
```bash
# Shakespeare 문자 수준 데이터셋 준비
cd nanoGPT/data/shakespeare_char
python prepare.py

# 생성되는 파일:
# - input.txt: 원본 텍스트 (~1MB)
# - train.bin: 학습 데이터 (1,003,854 토큰)
# - val.bin: 검증 데이터 (111,540 토큰)
# - meta.pkl: vocab 정보 (65개 문자 매핑)
```

#### 2. 데이터셋 및 Tokenizer 직접 살펴보기
```python
# Python으로 생성된 파일 확인하기
import pickle
import numpy as np

# 1. 원본 텍스트 확인
with open('input.txt', 'r') as f:
    text = f.read()
    print(text[:500])  # 첫 500자 출력

# 2. Meta 정보 확인 (tokenizer)
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
    print(f"Vocab size: {meta['vocab_size']}")
    print(f"Characters: {''.join(meta['itos'].values())}")

# 3. 인코딩된 데이터 확인
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
print(f"Train tokens: {len(train_data):,}")
print(f"First 20 tokens: {train_data[:20]}")

# 4. Tokenizer 테스트
encode = lambda s: [meta['stoi'][c] for c in s]
decode = lambda l: ''.join([meta['itos'][i] for i in l])

test_text = "ROMEO:"
encoded = encode(test_text)
decoded = decode(encoded)
print(f"Original: {test_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

#### 3. (선택) GPT-2 BPE Tokenizer 비교
```bash
# GPT-2 BPE를 사용한 Shakespeare 데이터셋 준비
cd ../shakespeare  # shakespeare (BPE 버전)
python prepare.py

# Character-level과 비교:
# - train.bin 크기 비교 (BPE가 더 작음)
# - 토큰 개수 비교 (BPE: 301,966 vs Char: 1,003,854)
```

```python
# tiktoken으로 BPE 토크나이저 테스트
import tiktoken

enc = tiktoken.get_encoding("gpt2")

test_text = "ROMEO:"
encoded = enc.encode(test_text)
decoded = enc.decode(encoded)

print(f"Original: {test_text}")
print(f"Encoded (BPE): {encoded}")  # 서브워드 단위 토큰 ID
print(f"Decoded: {decoded}")
print(f"Token count: {len(encoded)}")  # Character-level보다 짧음
```

#### 4. Tokenizer의 영향 관찰
- **문자 수준**: 한 글자씩 예측 → 세밀한 제어, 긴 시퀀스
- **BPE**: 서브워드 예측 → 효율적, 짧은 시퀀스, GPT-2 호환

---

## Task 3: 모델 구현 이해

### 참고할 파일
- **`nanoGPT/model.py`**: GPT 모델 전체 구현 (약 330줄)
- **`nanoGPT/config/train_shakespeare_char.py`**: 셰익스피어 학습용 설정

### 핵심 컴포넌트 분석 순서

#### 3.1 LayerNorm (line 18-27)
```python
# 목적: 학습 안정화
# 특징: bias 옵션 추가 (GPT-2와 호환)
```

#### 3.2 CausalSelfAttention (line 29-76) ⭐️ **가장 중요**
```python
# 핵심 개념:
# 1. Q, K, V 계산 (line 56)
# 2. Attention 점수 계산 (line 67)
# 3. Causal masking (line 68): 미래를 보지 못하게 함
# 4. Softmax + Weighted sum (line 69-71)
```

**주목할 점:**
- Flash Attention vs Manual Attention (line 62-71)
- 왜 `1.0 / math.sqrt(k.size(-1))`로 스케일링? (line 67)
- Causal mask가 없으면 어떻게 될까?

#### 3.3 MLP (line 78-92)
```python
# Feed-Forward Network
# n_embd -> 4*n_embd -> n_embd
# GELU 활성화 함수 사용
```

#### 3.4 Block (line 94-106)
```python
# Transformer 블록의 조합:
# x = x + Attention(LayerNorm(x))  # line 104
# x = x + MLP(LayerNorm(x))        # line 105
```

#### 3.5 GPT (line 118-330)
**중요 메서드:**
- `__init__` (line 120-148): 토큰/위치 임베딩, 레이어 스택
- `forward` (line 170-193): 순전파 및 손실 계산
- `generate` (line 306-330): 텍스트 생성 (autoregressive)

### 수행할 작업

1. **모델 파라미터 계산 이해**
   ```python
   # train_shakespeare_char.py 기본 설정:
   n_layer = 6      # Transformer 블록 개수
   n_head = 6       # Attention head 개수
   n_embd = 384     # 임베딩 차원
   block_size = 256 # 컨텍스트 길이
   vocab_size = 65  # 문자 개수

   # 총 파라미터: 약 10.7M
   ```

2. **Attention 시각화 준비**
   - Attention weights를 추출할 위치 파악 (line 70)
   - 학습 후 특정 문장의 attention map 그려보기

3. **코드 주석 달기**
   - `model.py`를 복사해서 자신의 이해를 주석으로 추가
   - 특히 `CausalSelfAttention.forward()` 메서드

---

## Task 4: 학습 수행

### 참고할 파일
- **`nanoGPT/train.py`**: 메인 학습 스크립트
- **`nanoGPT/config/train_shakespeare_char.py`**: 셰익스피어 학습 설정
- **`nanoGPT/configurator.py`**: 설정 오버라이드 시스템

### 학습 전략

#### 옵션 A: GPU 있는 경우 (추천)
```bash
# 기본 설정으로 학습 (~3분)
python train.py config/train_shakespeare_char.py

# 결과:
# - 학습 시간: ~3분 (A100 기준)
# - 최종 val loss: ~1.47
# - 체크포인트: out-shakespeare-char/ckpt.pt
```

#### 옵션 B: CPU만 있는 경우
```bash
# 작은 모델로 학습 (~3-5분)
python train.py config/train_shakespeare_char.py \
  --device=cpu \
  --compile=False \
  --eval_iters=20 \
  --block_size=64 \
  --batch_size=12 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --max_iters=2000 \
  --lr_decay_iters=2000 \
  --dropout=0.0

# 결과:
# - 학습 시간: ~3-5분
# - 최종 val loss: ~1.88 (더 높지만 학습 효과는 확인 가능)
```

### 학습 과정에서 관찰할 것

1. **Loss 감소 추이**
   ```
   iter 0: loss 4.2xxx  # 랜덤 초기화 (log(65) ≈ 4.17)
   iter 100: loss 2.8xxx
   iter 500: loss 1.9xxx
   iter 1000: loss 1.6xxx
   iter 2000: loss 1.5xxx  # 학습 완료
   ```

2. **왜 초기 loss가 ~4.2인가?**
   - 65개 문자를 랜덤 예측: -log(1/65) ≈ 4.17
   - **이것이 "전혀 학습 안된 상태"의 baseline!**

3. **MFU (Model FLOPs Utilization)**
   - GPU 활용률 모니터링
   - A100 기준 ~20-30%면 양호

### train.py 핵심 부분 이해

```python
# 학습 루프 (line 255-333):
while True:
    # 1. Learning rate 조정 (line 258-260)
    # 2. 평가 및 체크포인트 저장 (line 263-286)
    # 3. Forward pass (line 300)
    # 4. Backward pass (line 305)
    # 5. Gradient clipping (line 307-309)
    # 6. Optimizer step (line 311-312)
```

**주목할 함수:**
- `get_batch()` (line 116-131): 데이터 로딩
- `estimate_loss()` (line 216-228): 검증 손실 계산
- `get_lr()` (line 231-242): Cosine decay with warmup

### 수행할 작업

1. **학습 전 샘플 생성 (Baseline)**
   ```bash
   # 랜덤 초기화 모델로 생성
   python sample.py \
     --init_from=scratch \
     --start="ROMEO:" \
     --num_samples=3

   # 예상 결과: 완전 무작위 문자열
   ```

2. **학습 실행**
   ```bash
   python train.py config/train_shakespeare_char.py
   # 또는 CPU용 명령어
   ```

3. **학습 중 로그 관찰**
   - Loss가 감소하는지 확인
   - Train/Val loss gap 확인 (overfitting 여부)
   - 반복당 시간 확인

4. **학습 후 샘플 생성**
   ```bash
   python sample.py \
     --out_dir=out-shakespeare-char \
     --start="ROMEO:" \
     --num_samples=5 \
     --temperature=0.8
   ```

5. **Before/After 비교**
   - 학습 전: 무작위 문자
   - 학습 후: 셰익스피어 스타일 대사
   - **이것이 Attention 학습의 효과!**

---

## Task 5: 학습 효과 분석

### 참고할 파일
- **`nanoGPT/sample.py`**: 텍스트 생성 스크립트
- **`nanoGPT/model.py:306-330`**: generate 메서드

### 분석 항목

#### 5.1 정성적 분석: 생성 텍스트 품질

**체크리스트:**
- [ ] 올바른 문자 조합 (단어가 실제 영어처럼 보이는가?)
- [ ] 문법 구조 (주어-동사 순서가 맞는가?)
- [ ] 대화 형식 (캐릭터명: 대사 형식을 따르는가?)
- [ ] 문맥 유지 (몇 문장 동안 일관성이 있는가?)

**실험:**
```bash
# 다양한 temperature로 샘플링
python sample.py --out_dir=out-shakespeare-char --temperature=0.5  # 보수적
python sample.py --out_dir=out-shakespeare-char --temperature=1.0  # 균형
python sample.py --out_dir=out-shakespeare-char --temperature=1.5  # 창의적
```

#### 5.2 정량적 분석: Loss 비교

```python
# 분석할 값들:
# 1. 초기 loss: ~4.17 (random)
# 2. 최종 train loss: ~1.4-1.5
# 3. 최종 val loss: ~1.47-1.5
# 4. Loss 감소폭: ~2.7 (큰 개선!)
```

**Loss 의미:**
- Loss 4.17 → 2.0: 기본 문자 패턴 학습
- Loss 2.0 → 1.5: 단어 및 문법 패턴 학습
- Loss 1.5 이하: 문맥 및 스타일 학습

#### 5.3 Attention 패턴 분석 (심화)

**목표:** Attention이 실제로 무엇을 학습했는지 시각화

**방법:**
1. `model.py`의 `CausalSelfAttention.forward()` 수정
2. Attention weights 저장 (line 69의 `att` 변수)
3. 특정 문장 입력 시 attention map 그리기

**예상 패턴:**
- 대명사 → 이전 명사 attend
- 동사 → 주어 attend
- 형용사 → 명사 attend
- 문장 끝 → 문장 시작 attend

**구현 예시 (선택사항):**
```python
# sample.py 수정하여 attention weights 추출
# matplotlib으로 heatmap 그리기
```

---

## 학습 로드맵 요약

| Task | 파일 | 소요 시간 | 목표 |
|------|------|----------|------|
| **1. 목표 설정** | README.md, model.py | 30분 | Attention 메커니즘 이해 |
| **2. 데이터 준비** | data/shakespeare_char/prepare.py | 10분 | 데이터셋 구조 파악 |
| **3. 모델 이해** | model.py, config/train_shakespeare_char.py | 1-2시간 | GPT 아키텍처 분석 |
| **4. 학습 수행** | train.py, sample.py | 3-5분 (학습)<br>10분 (실험) | Before/After 비교 |
| **5. 효과 분석** | sample.py, model.py | 30분-1시간 | Attention 학습 효과 확인 |

**총 소요 시간:** 약 3-5시간 (학습 자체는 3-5분)

---

## 추천 학습 순서

### Day 1: 이론 및 코드 이해
1. `README.md` 읽기 (Quick Start 섹션)
2. `model.py` 읽기 (특히 CausalSelfAttention)
3. 데이터 준비: `python data/shakespeare_char/prepare.py`
4. 데이터 구조 파악

### Day 2: 실습 및 분석
1. **학습 전 샘플 생성** (랜덤 모델)
2. **학습 실행** (3-5분)
3. **학습 후 샘플 생성** (학습된 모델)
4. **Before/After 비교 분석**
5. 다양한 temperature 실험

### Day 3: 심화 (선택)
1. Attention weights 추출 및 시각화
2. 더 긴 학습 (max_iters 증가)
3. 모델 크기 변경 실험
4. 다른 데이터셋 시도

---

## 핵심 파일 체크리스트

학습 과정에서 **반드시 읽어야 할 파일:**

- [ ] `nanoGPT/README.md` - Quick Start 섹션
- [ ] `nanoGPT/model.py` - 특히 CausalSelfAttention 클래스
- [ ] `nanoGPT/data/shakespeare_char/prepare.py` - 데이터 전처리
- [ ] `nanoGPT/config/train_shakespeare_char.py` - 학습 설정
- [ ] `nanoGPT/train.py` - line 116-131 (get_batch), line 255-333 (학습 루프)
- [ ] `nanoGPT/sample.py` - line 83-89 (생성 루프)

**선택적으로 읽을 파일:**

- [ ] `nanoGPT/configurator.py` - 설정 시스템 이해
- [ ] `nanoGPT/bench.py` - 성능 벤치마킹
- [ ] `nanoGPT/config/train_gpt2.py` - 대규모 학습 설정 참고

---

## 기대 효과 및 학습 성과

### 학습 전 (랜덤 초기화)
```
ROMEO:
xK3!zQp$mWvL&
```
- Loss: ~4.17
- 완전 무작위 문자
- Attention이 아무것도 학습 안함

### 학습 후 (3분 학습)
```
ROMEO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.
```
- Loss: ~1.47
- 셰익스피어 스타일 문법
- **Attention이 문맥 의존성을 학습!**

### 핵심 깨달음
1. **Attention의 역할:** 문맥 정보를 선택적으로 가져옴
2. **학습의 의미:** 어떤 단어가 어떤 단어를 참조해야 하는지 학습
3. **NLP 능력 획득:** 문법, 문맥, 스타일을 자연스럽게 생성

---

## 추가 실험 아이디어

1. **컨텍스트 길이 실험**
   ```bash
   # block_size를 바꿔가며 학습
   python train.py config/train_shakespeare_char.py --block_size=64
   python train.py config/train_shakespeare_char.py --block_size=256
   python train.py config/train_shakespeare_char.py --block_size=512
   ```

2. **모델 크기 실험**
   ```bash
   # 작은 모델
   python train.py config/train_shakespeare_char.py --n_layer=2 --n_head=2 --n_embd=128

   # 큰 모델
   python train.py config/train_shakespeare_char.py --n_layer=12 --n_head=12 --n_embd=768
   ```

3. **Attention Head 수 실험**
   ```bash
   # Single-head attention
   python train.py config/train_shakespeare_char.py --n_head=1

   # Multi-head attention
   python train.py config/train_shakespeare_char.py --n_head=6
   ```

4. **학습 데이터 양 실험**
   - `prepare.py` 수정하여 10%, 50%, 100% 데이터로 학습
   - 데이터 양에 따른 성능 변화 관찰

---

## 문제 해결

### GPU 메모리 부족
```bash
# batch_size 줄이기
python train.py config/train_shakespeare_char.py --batch_size=8

# 모델 크기 줄이기
python train.py config/train_shakespeare_char.py --n_layer=4 --n_embd=256
```

### CPU에서 너무 느림
```bash
# 반복 횟수 줄이기
python train.py config/train_shakespeare_char.py --max_iters=500

# 평가 빈도 줄이기
python train.py config/train_shakespeare_char.py --eval_interval=100
```

### PyTorch 2.0 compile 에러
```bash
# compile 비활성화
python train.py config/train_shakespeare_char.py --compile=False
```

---

## 참고 자료

- [Andrej Karpathy's YouTube: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention Is All You Need (논문)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [nanoGPT GitHub Repository](https://github.com/karpathy/nanoGPT)
