# torch_ckpt — PyTorch 체크포인트 매니저

PyTorch 딥러닝 실험의 재현성과 추적 가능성을 보존하기 위한 체크포인트 관리 모듈입니다.

---

## 프로젝트 개요

딥러닝 실험을 진행하다 보면 다음과 같은 문제가 자주 발생합니다.

- 학습을 완료했으나 어떤 설정으로 실행했는지 기억나지 않는 상황
- 체크포인트를 불러와도 정확히 동일한 결과가 재현되지 않는 문제
- GPU 사용량, OS, Python/CUDA 버전 등 실행 환경 정보가 사후에 소실되는 문제
- 모델 가중치만 저장하고 optimizer state 또는 scheduler state를 누락하는 문제

`torch_ckpt`는 이러한 문제들을 하나의 체크포인트 파일로 해결하기 위해 설계된 모듈입니다.

**두 가지 핵심 목표:**

1. **재현성(Reproducibility)** — 실험 재현에 필요한 모든 요소(시드 상태, Git 커밋, 하이퍼파라미터, 환경 정보)를 빠짐없이 설정하여 테스트하고 저장합니다.
2. **편의성** — 모든 구성 요소를 하나의 딕셔너리로 관리하여 실험 추적을 단순화합니다.

---

## 모듈 구조

```
utils/
├── torch_ckpt.py          # ckpt_manager 클래스 (메인)
└── ckpt_modules/
    ├── __init__.py
    └── settings.py        # 6개의 Settings 클래스
```

`ckpt_manager`는 아래 6개의 Settings 클래스를 조합하여 동작합니다.

| 클래스 | 역할 |
| --- | --- |
| `SeedSettings` | 랜덤 시드 및 결정론적 모드 관리 |
| `DLSettings` | 모델, 옵티마이저, 스케줄러 인스턴스 및 하이퍼파라미터 관리 |
| `PathSettings` | 체크포인트 저장 경로 및 사용자 메모 관리 |
| `GitSettings` | 실험 시점의 Git 커밋 해시, 브랜치, dirty 상태 추적 |
| `EnvSettings` | 연산 장치, 코드 스냅샷, requirements.txt 저장 설정 |
| `TimeSettings` | 초기화 시점의 타임스탬프 자동 기록 |

---

## 입력: config.json 구성

`ckpt_manager`는 `config.json` 파일을 로드하여 초기화하는 패턴을 사용합니다. 아래는 전체 config.json 예시입니다.

```json
{
  "proj_dir": "/path/to/project",
  "seed_settings": {
    "use_seed": true,
    "seed": 42,
    "use_deterministic": true
  },
  "deep_learning_settings": {
    "data_config": {"batch_size": 64},
    "model_config": {"embed_dim": 512, "n_head": 8},
    "optimizer_config": {"lr": 3e-4},
    "trainer_config": {"max_iters": 10000, "iter_eval": 500}
  },
  "path_settings": {
    "file_path": "/path/to/Trainer.py",
    "save_path": "/path/to/checkpoints",
    "save_name": "best_model.pt",
    "user_note": "baseline experiment"
  },
  "git_settings": {
    "use_git": true,
    "strict_git": false
  },
  "env_settings": {
    "device": "cuda",
    "save_code": true,
    "save_requirements_txt": true
  }
}
```

### seed_settings

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `use_seed` | `bool` | 시드 적용 여부 |
| `seed` | `int` | 기본 시드 값 (예: `42`) |
| `use_deterministic` | `bool` | `torch.use_deterministic_algorithms(True)` 활성화 여부. `True`로 설정 시 환경 변수 `CUBLAS_WORKSPACE_CONFIG=:4096:8`도 자동으로 설정됩니다. |

### deep_learning_settings

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `data_config` | `dict` | 데이터 설정 (batch_size, 경로 등) |
| `model_config` | `dict` | 모델 하이퍼파라미터 (embed_dim, n_head 등) |
| `optimizer_config` | `dict` | 옵티마이저 설정 (lr 등) |
| `trainer_config` | `dict` | 트레이너 설정 (max_iters, iter_eval 등) |

모델, 옵티마이저, 스케줄러 인스턴스(`model`, `optimizer`, `scheduler`)는 초기화 시점에 주입할 필요 없이, 체크포인트 저장 전에 할당하면 됩니다.

### path_settings

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `file_path` | `str` | 실행 파일(.py/.ipynb) 경로 |
| `save_path` | `str` | 체크포인트 저장 디렉토리 |
| `save_name` | `str` | 저장할 `.pt` 파일 이름 |
| `user_note` | `str` | 자유 형식 메모 (실험 설명 등) |

> **주의:** Jupyter Notebook에서 실행하실 경우, `file_path`는 자동 탐지가 불가능하므로 반드시 직접 입력해 주셔야 합니다.

### git_settings

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `use_git` | `bool` | Git 추적 활성화 여부 |
| `strict_git` | `bool` | `True`로 설정하면 미커밋 변경사항이 있을 때 에러가 발생합니다. `False`(기본값)이면 경고 메시지를 출력하고 계속 진행합니다. |

### env_settings

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `device` | `str` | 연산 장치 (`"cuda"` 또는 `"cpu"`) |
| `save_code` | `bool` | 코드 스냅샷 저장 여부 |
| `save_requirements_txt` | `bool` | `pip freeze` 결과를 `requirements.txt`로 저장할지 여부 |

---

## 기술 스택

| 분류 | 기술 |
| --- | --- |
| 언어 | Python 3.x |
| 딥러닝 | PyTorch |
| 시스템 모니터링 | psutil |
| Git 추적 | subprocess (git 명령어) |
| 환경 스냅샷 | pip freeze |

---

## 사용 방법

`config.json`을 기반으로 초기화하고 학습 루프 안에서 체크포인트를 저장하는 전형적인 패턴입니다.

```python
import json
from utils import torch_ckpt

# 1. config.json 로드
with open("config.json", "r") as f:
    config = json.load(f)

# 2. ckpt_manager 초기화 (시드 적용, requirements.txt 저장 등 자동 수행)
ckpt = torch_ckpt.ckpt_manager(**config)

# 3. 모델/옵티마이저 생성 후 주입
ckpt.deep_learning_settings.model = gpt_model
ckpt.deep_learning_settings.optimizer = optimizer

# 4. 학습 루프에서 체크포인트 저장
for epoch in range(max_epochs):
    train_loss = train_one_epoch(...)
    val_loss   = validate(...)

    is_best = val_loss < best_val_loss
    ckpt.save_ckpt(
        epoch=epoch,
        step=global_step,
        train_loss_history=train_losses,
        val_loss_history=val_losses,
        best_val_loss=best_val_loss,
        patience_counter=patience,
        model_save=is_best,   # best 모델일 때만 가중치 저장
        save_all_env=False,
    )
```

### model_save 옵션

| 옵션 | 저장 내용 | 용도 |
| --- | --- | --- |
| `model_save=True` | 메타정보 + 모델/옵티마이저/스케줄러 가중치 전체 | best 모델 저장 시 사용 |
| `model_save=False` | 메타정보만 저장 (가중치 제외) | 매 iteration 로그 기록 시 사용. 디스크를 약 90% 절약할 수 있습니다. |

매 iteration마다 `model_save=False`로 로그를 남기고, best 모델일 때만 `model_save=True`를 사용하는 패턴을 권장합니다.

### strict_git 옵션

| 값 | 동작 |
| --- | --- |
| `strict_git=False` (기본값) | 미커밋 변경사항이 있어도 경고 메시지만 출력하고 계속 진행합니다. 탐색 단계의 실험에 적합합니다. |
| `strict_git=True` | 미커밋 변경사항이 있으면 에러를 발생시킵니다. 공식 실험에서 재현성을 강하게 보장할 때 권장합니다. |

---

## 최종 출력: 체크포인트 파일 구조

체크포인트는 `path_settings.save_path / save_name` 경로에 저장됩니다 (예: `checkpoints/best_model.pt`).

`torch.load()`로 불러오면 아래 구조의 딕셔너리를 반환합니다.

```
checkpoint.pt
│
├── epoch, step
├── epoch_loss
│   ├── train_loss        (list)
│   ├── val_loss          (list)
│   ├── best_val_loss
│   └── patience_counter
│
├── seed
│   ├── seed
│   ├── random_state      (random 모듈 상태)
│   ├── numpy_state       (numpy RNG 상태)
│   ├── torch_state       (PyTorch CPU RNG 상태)
│   └── cuda_state        (PyTorch CUDA RNG 상태)
│
├── git
│   ├── commit_hash
│   ├── branch
│   └── is_dirty
│
├── config                (전체 하이퍼파라미터)
├── cli_args
├── training_time         (초 단위)
├── executed_datetime
│
├── env                   (Python/PyTorch/CUDA 버전)
├── system_info
│   ├── cpu
│   ├── memory
│   └── gpu_memory        (current + peak VRAM)
│
├── user_notes
│
└── (model_save=True 일 때만)
    ├── model_state_dict
    ├── optimizer_state_dict
    └── scheduler_state_dict
```

**저장 카테고리별 상세 설명:**

- **학습 상태**: `epoch`, `step`, 손실 이력(`train_loss`, `val_loss`), `best_val_loss`, `patience_counter`를 저장합니다.
- **재현성 정보**: 시드 설정 값 및 저장 시점의 RNG 상태(random, numpy, PyTorch CPU/CUDA), Git 커밋 해시/브랜치/dirty 여부, 전체 하이퍼파라미터(`config`), CLI 인자(`cli_args`)를 저장합니다.
- **환경 정보**: Python/PyTorch/CUDA 버전, GPU 이름, OS 정보(`env`), CPU 스펙 및 사용률, RAM 사용량, GPU VRAM 현재값 및 피크값(`system_info`), 학습 경과 시간(`training_time`), 시작/종료 시각(`executed_datetime`), 사용자 메모(`user_notes`)를 저장합니다.

> **참고:** GPU 메모리는 체크포인트 저장 시점에 낮아져 있을 수 있습니다. 이 때문에 `ckpt_manager` 초기화 시 `reset_peak_memory_stats()`를 호출하고, 저장 시점에 `max_memory_allocated()`로 학습 중 최대 VRAM 사용량(`peak_allocated_gb`)을 별도로 기록합니다.

`model_save=False`로 저장하시면 가중치 없이 메타정보만 저장되어 디스크를 약 90% 절약할 수 있습니다.

---

## 설치

```bash
pip install -r requirements.txt
```
