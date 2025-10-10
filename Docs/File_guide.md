# nanoGPT 프로젝트 파일 가이드

## 1. config/ 폴더

`config/` 폴더는 다양한 학습 및 평가 시나리오를 위한 설정 파일들을 포함합니다. 이 파일들은 `configurator.py`에 의해 로드되며, `train.py`나 다른 스크립트 실행 시 명령줄 인자로 전달됩니다.

### 1.1 eval_*.py (평가 설정 파일)

사전 훈련된 GPT-2 모델의 성능을 평가하기 위한 설정 파일들입니다.

**파일 목록:**
- `eval_gpt2.py` - GPT-2 base 모델 (124M 파라미터)
- `eval_gpt2_medium.py` - GPT-2 medium 모델
- `eval_gpt2_large.py` - GPT-2 large 모델
- `eval_gpt2_xl.py` - GPT-2 XL 모델

**주요 설정 예시 (eval_gpt2.py):**
```python
batch_size = 8
eval_iters = 500  # 정확한 평가를 위해 더 많은 반복 사용
eval_only = True  # 평가만 수행하고 종료
wandb_log = False
init_from = 'gpt2'  # 사전 훈련된 GPT-2 모델 사용
```

**용도:**
- 사전 훈련된 OpenAI GPT-2 모델의 검증 손실(validation loss)을 측정
- 다른 데이터셋에서 GPT-2 모델의 성능 벤치마킹

### 1.2 finetune_*.py (파인튜닝 설정 파일)

사전 훈련된 모델을 특정 데이터셋에 맞게 미세 조정하기 위한 설정 파일들입니다.

**파일 목록:**
- `finetune_shakespeare.py` - Shakespeare 데이터셋 파인튜닝

**주요 설정 예시 (finetune_shakespeare.py):**
```python
out_dir = 'out-shakespeare'
dataset = 'shakespeare'
init_from = 'gpt2-xl'  # 가장 큰 GPT-2 모델에서 시작
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20
learning_rate = 3e-5  # 파인튜닝을 위한 낮은 학습률
decay_lr = False  # 일정한 학습률 사용
always_save_checkpoint = False  # 검증 손실이 개선될 때만 저장
```

**특징:**
- 사전 훈련된 모델(`init_from`)에서 시작
- 작은 데이터셋에 맞춰 낮은 학습률과 적은 반복 횟수 사용
- Shakespeare 데이터셋은 301,966개의 토큰으로 구성되어 1 에폭 ≈ 9.2 반복

### 1.3 train_*.py (학습 설정 파일)

모델을 처음부터 또는 특정 설정으로 학습시키기 위한 설정 파일들입니다.

**파일 목록:**
- `train_gpt2.py` - GPT-2 (124M) 모델 학습 설정
- `train_shakespeare_char.py` - Shakespeare 문자 레벨 모델 학습

**주요 설정 예시 (train_gpt2.py):**
```python
# 8개의 A100 40GB GPU로 약 5일간 학습하여 ~2.85 손실 달성
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8  # 총 배치 크기 ~0.5M 토큰
max_iters = 600000  # 총 300B 토큰 학습
weight_decay = 1e-1
```

**주요 설정 예시 (train_shakespeare_char.py):**
```python
# 작은 문자 레벨 모델 (디버깅 및 실험용)
out_dir = 'out-shakespeare-char'
dataset = 'shakespeare_char'
batch_size = 64
block_size = 256  # 최대 256개 이전 문자 컨텍스트

# Baby GPT 모델 구조
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # 작은 모델이므로 더 높은 학습률 사용 가능
max_iters = 5000
```

**특징:**
- `train_gpt2.py`: 대규모 학습을 위한 설정 (멀티 GPU, 높은 배치 크기)
- `train_shakespeare_char.py`: 작은 모델로 빠른 실험 및 디버깅용 (맥북에서도 실행 가능)

---

## 2. .gitattributes

`.gitattributes` 파일은 Git이 특정 파일을 어떻게 처리할지 지정하는 설정 파일입니다. `.gitignore`가 파일 추적 여부를 결정한다면, `.gitattributes`는 추적되는 파일의 속성을 제어합니다.

**현재 내용:**
```
# Override jupyter in Github language stats for more accurate estimate of repo code languages
# reference: https://github.com/github/linguist/blob/master/docs/overrides.md#generated-code
*.ipynb linguist-generated
```

**설명:**
- `*.ipynb linguist-generated`: 모든 Jupyter Notebook 파일(`.ipynb`)을 "생성된 코드"로 표시
- GitHub의 언어 통계(language statistics)에서 Jupyter Notebook 파일을 제외
- 이를 통해 저장소의 주요 프로그래밍 언어가 Python 코드로 더 정확하게 표시됨

**일반적인 .gitattributes 용도:**
- 텍스트 파일의 줄바꿈 형식 지정 (CRLF vs LF)
- 파일 병합 전략 설정
- 파일을 바이너리로 처리하도록 지정
- 언어 통계 조정
- Git diff 동작 커스터마이징

---

## 3. LICENSE

MIT License 파일로, 이 프로젝트의 사용 및 배포 조건을 명시합니다.

**저작권:**
- Copyright (c) 2022 Andrej Karpathy

**주요 내용:**
- 누구나 무료로 소프트웨어를 사용, 복사, 수정, 병합, 게시, 배포, 재라이선스, 판매할 수 있음
- 저작권 표시와 라이선스 고지를 모든 복사본에 포함해야 함
- 소프트웨어는 "있는 그대로" 제공되며, 어떠한 보증도 없음
- 저작자는 소프트웨어 사용으로 인한 책임을 지지 않음

**MIT 라이선스의 특징:**
- 가장 허용적인(permissive) 오픈소스 라이선스 중 하나
- 상업적 사용 가능
- 수정 및 재배포 자유
- 최소한의 제약 조건

---

## 4. bench.py (벤치마킹 스크립트)

모델의 학습 성능을 빠르게 측정하기 위한 간소화된 벤치마킹 도구입니다.

**주요 기능:**
1. **모델 초기화**
   - GPT-2 크기 모델 생성 (12 layers, 12 heads, 768 embedding dimension)
   - 약 124M 파라미터

2. **데이터 로딩**
   - `real_data=True`: OpenWebText 데이터셋 사용
   - `real_data=False`: 랜덤 데이터 사용 (데이터 로딩 오버헤드 제거)

3. **벤치마킹 옵션**
   - **단순 벤치마킹**: 10회 워밍업 후 20회 반복으로 성능 측정
   - **프로파일링 모드**: PyTorch Profiler로 상세 성능 분석

4. **성능 지표**
   - 반복당 시간 (ms)
   - MFU (Model FLOPs Utilization): A100 GPU 대비 모델 FLOPS 활용률

**주요 설정:**
```python
batch_size = 12
block_size = 1024
device = 'cuda'
dtype = 'bfloat16' or 'float16'
compile = True  # PyTorch 2.0 컴파일 사용
```

**사용 예시:**
```bash
python bench.py
python bench.py --batch_size=32 --compile=False
```

---

## 5. configurator.py (설정 오버라이드 도구)

"Poor Man's Configurator"라고 불리는 간단한 설정 오버라이드 시스템입니다.

**작동 방식:**
1. `train.py` 등에서 `exec(open('configurator.py').read())`로 실행됨
2. 명령줄 인자를 파싱하여 전역 변수를 오버라이드

**사용 방법:**

1. **설정 파일 사용:**
```bash
python train.py config/train_gpt2.py
```
- `config/train_gpt2.py` 파일의 내용을 실행하여 변수 오버라이드

2. **명령줄 인자 사용:**
```bash
python train.py --batch_size=32 --learning_rate=1e-4
```
- 기존 전역 변수의 값을 명령줄에서 직접 오버라이드

3. **조합 사용:**
```bash
python train.py config/train_gpt2.py --batch_size=32
```
- 먼저 설정 파일 적용 후, 명령줄 인자로 추가 오버라이드

**기능:**
- `literal_eval`을 사용하여 타입 자동 변환 (숫자, 불린 등)
- 기존 변수의 타입과 일치하는지 검증
- 존재하지 않는 설정 키에 대해 오류 발생

---

## 6. model.py (GPT 모델 정의)

GPT 언어 모델의 전체 아키텍처를 포함한 핵심 파일입니다.

### 6.1 주요 클래스

#### LayerNorm
- PyTorch의 LayerNorm이지만 bias를 선택적으로 사용할 수 있는 버전

#### CausalSelfAttention
- **멀티헤드 인과적 자기 어텐션(Multi-head Causal Self-Attention)**
- Query, Key, Value 프로젝션을 배치로 처리
- Flash Attention 지원 (PyTorch >= 2.0)
- 인과적 마스킹으로 미래 토큰을 볼 수 없도록 제한

#### MLP (Multi-Layer Perceptron)
- 피드포워드 네트워크
- 구조: Linear → GELU → Linear → Dropout
- 숨겨진 차원은 `4 * n_embd`

#### Block
- Transformer 블록의 기본 단위
- 구조: LayerNorm → Attention → Residual Connection → LayerNorm → MLP → Residual Connection

#### GPTConfig
- 모델 설정을 위한 데이터클래스
- 주요 파라미터:
  - `block_size`: 최대 시퀀스 길이 (기본 1024)
  - `vocab_size`: 어휘 크기 (기본 50304)
  - `n_layer`: 레이어 수 (기본 12)
  - `n_head`: 어텐션 헤드 수 (기본 12)
  - `n_embd`: 임베딩 차원 (기본 768)
  - `dropout`: 드롭아웃 비율 (기본 0.0)
  - `bias`: Linear와 LayerNorm에 bias 사용 여부

### 6.2 GPT 클래스

**주요 메서드:**

1. **`__init__(config)`**
   - 토큰 임베딩 (`wte`)과 위치 임베딩 (`wpe`) 생성
   - Transformer 블록들 스택 생성
   - Language modeling head 생성
   - Weight tying: `wte`와 `lm_head`의 가중치 공유

2. **`forward(idx, targets=None)`**
   - 순전파 함수
   - 토큰과 위치 임베딩 합산
   - Transformer 블록들 통과
   - 로짓 계산 및 손실 계산 (targets가 주어진 경우)

3. **`from_pretrained(model_type, override_args=None)`**
   - OpenAI의 사전 훈련된 GPT-2 모델 로드
   - 지원 모델: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
   - Hugging Face transformers에서 가중치를 가져와 변환

4. **`configure_optimizers(weight_decay, learning_rate, betas, device_type)`**
   - AdamW 옵티마이저 설정
   - 2D 파라미터(가중치)에만 weight decay 적용
   - 1D 파라미터(bias, LayerNorm)에는 weight decay 미적용
   - CUDA에서 fused AdamW 사용 가능 시 자동 활성화

5. **`generate(idx, max_new_tokens, temperature=1.0, top_k=None)`**
   - 자동회귀 텍스트 생성
   - Temperature sampling으로 다양성 조절
   - Top-k sampling 지원

6. **`estimate_mfu(fwdbwd_per_iter, dt)`**
   - Model FLOPs Utilization (MFU) 추정
   - A100 GPU의 이론적 최대 성능 대비 실제 활용률 계산

7. **`crop_block_size(block_size)`**
   - 모델의 컨텍스트 크기를 동적으로 축소
   - 위치 임베딩과 어텐션 마스크 조정

### 6.3 주요 특징

- **Weight Tying**: 토큰 임베딩과 출력 레이어 가중치 공유로 파라미터 효율성 향상
- **Flash Attention**: PyTorch 2.0 이상에서 자동으로 고속 어텐션 사용
- **특수한 초기화**: Residual projection에 대해 GPT-2 논문의 scaled initialization 적용
- **Flexible Bias**: LayerNorm과 Linear 레이어에서 bias 사용 여부를 선택 가능

---

## 7. sample.py (텍스트 생성 스크립트)

학습된 모델을 사용하여 텍스트를 생성하는 스크립트입니다.

**주요 기능:**

1. **모델 로딩**
   - `init_from='resume'`: 특정 디렉토리에서 체크포인트 로드
   - `init_from='gpt2'` (또는 다른 variant): OpenAI GPT-2 사전 훈련 모델 로드

2. **인코딩/디코딩**
   - 데이터셋의 `meta.pkl` 파일에서 커스텀 인코더 로드
   - 없으면 GPT-2 tiktoken 인코더 사용

3. **텍스트 생성**
   - 시작 프롬프트에서 시작하여 자동회귀 방식으로 생성
   - Temperature와 top-k sampling으로 생성 품질 조절

**주요 설정:**
```python
init_from = 'resume'  # 또는 'gpt2-xl' 등
out_dir = 'out'  # 체크포인트 디렉토리
start = "\n"  # 또는 "FILE:prompt.txt"
num_samples = 10  # 생성할 샘플 수
max_new_tokens = 500  # 샘플당 생성할 토큰 수
temperature = 0.8  # 낮을수록 결정적, 높을수록 무작위
top_k = 200  # 상위 k개 토큰만 고려
```

**사용 예시:**
```bash
# 학습된 모델에서 샘플 생성
python sample.py --init_from=resume --out_dir=out-shakespeare

# GPT-2 모델에서 샘플 생성
python sample.py --init_from=gpt2-xl --start="Once upon a time"

# 파일에서 프롬프트 읽기
python sample.py --start="FILE:prompt.txt"

# 설정 조정
python sample.py --num_samples=5 --max_new_tokens=1000 --temperature=1.0
```

**프롬프트 옵션:**
- 직접 문자열: `start="Once upon a time"`
- 파일에서 읽기: `start="FILE:prompt.txt"`
- 특수 토큰: `start="<|endoftext|>"`

---

## 8. train.py (학습 메인 스크립트)

GPT 모델을 학습시키는 핵심 스크립트입니다. 단일 GPU와 분산 데이터 병렬(DDP) 모두 지원합니다.

### 8.1 주요 기능

1. **다양한 초기화 모드**
   - `scratch`: 처음부터 새로운 모델 학습
   - `resume`: 이전 체크포인트에서 학습 재개
   - `gpt2*`: OpenAI GPT-2 모델에서 시작 (파인튜닝)

2. **분산 학습 (DDP)**
   - 여러 GPU와 노드에서 병렬 학습 지원
   - `torchrun`을 사용한 간편한 실행

3. **그래디언트 누적**
   - 메모리 제약 하에서 큰 배치 크기 시뮬레이션

4. **학습률 스케줄링**
   - 워밍업 단계
   - 코사인 감쇠 (Cosine decay)

5. **평가 및 체크포인팅**
   - 주기적으로 train/validation 손실 평가
   - 최고 성능 모델 자동 저장

6. **Weights & Biases 통합**
   - 학습 메트릭 로깅 및 시각화

### 8.2 주요 설정

**I/O 설정:**
```python
out_dir = 'out'  # 체크포인트 저장 디렉토리
eval_interval = 2000  # 평가 주기
log_interval = 1  # 로깅 주기
eval_iters = 200  # 평가 시 사용할 반복 횟수
```

**데이터 설정:**
```python
dataset = 'openwebtext'
gradient_accumulation_steps = 40  # 큰 배치 크기 시뮬레이션
batch_size = 12  # 마이크로 배치 크기
block_size = 1024  # 컨텍스트 길이
```

**모델 설정:**
```python
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
```

**옵티마이저 설정:**
```python
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
```

**학습률 감쇠 설정:**
```python
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
```

### 8.3 사용 예시

**단일 GPU 학습:**
```bash
python train.py --batch_size=32 --compile=False
```

**단일 노드, 4 GPU DDP:**
```bash
torchrun --standalone --nproc_per_node=4 train.py
```

**다중 노드 DDP (2 노드, 각 8 GPU):**
```bash
# 마스터 노드
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=123.456.123.456 --master_port=1234 train.py

# 워커 노드
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=123.456.123.456 --master_port=1234 train.py
```

**설정 파일 사용:**
```bash
python train.py config/train_gpt2.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

### 8.4 학습 루프

1. **학습률 조정**: 워밍업 또는 코사인 감쇠 적용
2. **평가**: 주기적으로 train/val 손실 계산
3. **체크포인팅**: 최고 성능 시 모델 저장
4. **순전파/역전파**: 그래디언트 누적과 함께 수행
5. **그래디언트 클리핑**: 안정적인 학습을 위해 적용
6. **옵티마이저 스텝**: 가중치 업데이트
7. **로깅**: 손실, 시간, MFU 등 출력

### 8.5 주요 함수

**`get_batch(split)`**
- train 또는 val 데이터에서 배치 샘플링
- numpy memmap으로 메모리 효율적 데이터 로딩

**`estimate_loss()`**
- 여러 배치에 걸쳐 정확한 손실 추정
- 평가 모드에서 실행

**`get_lr(it)`**
- 현재 반복에 따른 학습률 계산
- 워밍업 → 코사인 감쇠 → 최소 학습률

---

## 파일 요약

| 파일/폴더 | 용도 | 중요도 |
|----------|------|--------|
| `config/` | 다양한 학습/평가 시나리오 설정 | ⭐⭐⭐ |
| `model.py` | GPT 모델 아키텍처 정의 | ⭐⭐⭐⭐⭐ |
| `train.py` | 메인 학습 스크립트 | ⭐⭐⭐⭐⭐ |
| `sample.py` | 텍스트 생성 스크립트 | ⭐⭐⭐⭐ |
| `configurator.py` | 설정 오버라이드 시스템 | ⭐⭐⭐ |
| `bench.py` | 성능 벤치마킹 도구 | ⭐⭐⭐ |
| `.gitattributes` | Git 파일 속성 설정 | ⭐ |
| `LICENSE` | 라이선스 정보 (MIT) | ⭐ |

---

## 학습 워크플로우 예시

### 1. 작은 모델로 실험 (맥북/로컬)
```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
python sample.py --out_dir=out-shakespeare-char --start="ROMEO:"
```

### 2. GPT-2 모델 평가
```bash
python train.py config/eval_gpt2.py
```

### 3. Shakespeare 파인튜닝
```bash
python train.py config/finetune_shakespeare.py
```

### 4. 대규모 GPT-2 학습
```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

### 5. 성능 벤치마킹
```bash
python bench.py --device=cuda --compile=True
```
