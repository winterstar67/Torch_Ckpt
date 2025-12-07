'''
2025-11-23
Make kind comment for each option or code line


### 2025 11-25
Purpose of this module: the main focus is two things 
1. Ensuring reproduction when you do experiments ensuring you didn't miss any component save 
2. We can manage the environment that I run conveniently ensuring all components are in one dictionary - which makes super convenient

What're not implemented yet?
- Data Info part
    - At which batch position did it saved?
    - What's the batch size, dimension, example data?
    - 1) Dataloader / sampler RNG state (important)
        - If you use DataLoader with num_workers > 0 and any random augmentation / shuffling:
        - Every worker has its own RNG state.
        - Just saving global torch.get_rng_state() and cuda_state is not enough to reproduce the exact minibatch order + augmentations.
- 

- In the case of Multi-GPU / DDP-specific state, current system could be not worked


'''

import shutil
import os
import time
import random
import subprocess
from datetime import datetime
from pathlib import Path
import platform

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Any, List, Union

import sys

# Import all settings classes from the ckpt_modules package
from .ckpt_modules import (
    SeedSettings,
    DLSettings,
    PathSettings,
    GitSettings,
    EnvSettings,
    TimeSettings,
)


class ckpt_manager:
    """Checkpoint manager for tracking training state and reproducibility."""

    def __init__(
        self,
        seed_settings: dict,
        deep_learning_settings: Union[DLSettings, dict],
        path_settings: Union[PathSettings, dict],
        git_settings: Union[GitSettings, dict],
        env_settings: Union[EnvSettings, dict]
    ):
        """
        Initialize checkpoint manager.

        Args:
            seed_settings: SeedSettings instance or dict with seed configuration
            deep_learning_settings: DLSettings instance or dict with DL configuration
            path_settings: PathSettings instance or dict with path configuration
            git_settings: GitSettings instance or dict with git configuration
            env_settings: EnvSettings instance or dict with environment configuration
        """
        # Store settings
        self.seed_settings = SeedSettings(seed_settings)
        self.deep_learning_settings = DLSettings(deep_learning_settings)
        self.path_settings = PathSettings(path_settings)
        self.git_settings = GitSettings(git_settings)
        self.env_settings = EnvSettings(env_settings)

        # Initialize time settings
        self.time_settings = TimeSettings()

        # Apply seed settings
        self.set_seed(self.seed_settings.seed)
        if self.seed_settings.use_deterministic:
            torch.use_deterministic_algorithms(True)

        # Save requirements if requested
        if self.env_settings.save_requirements_txt:
            self.save_requirements(self.path_settings.save_path)

        # Reset GPU memory stats to track peak from training start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def save_code_snapshot(self, src, dst):
        assert (src[-3:] == ".py") or (src[-6:] == ".ipynb"), "You can save .py or .ipynb"

        shutil.copy(src, dst)

    def save_requirements(self, save_dir: str):
        """Create requirements.txt using pip freeze."""
        req_path = os.path.join(save_dir, "requirements.txt")
        try:
            requirements = subprocess.getoutput("pip freeze")
            with open(req_path, "w") as f:
                f.write(requirements)
        except Exception as e:
            print(f"[Warning] Failed to save requirements.txt: {e}")

    # === Seed config ===
    def set_seed(self, seed: int):
        '''
        Set seed for reproducibility
        '''
        random.seed(seed) # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # === System Information Tracking ===
    def get_cpu_info(self) -> Dict[str, Any]:
        """
        Get detailed CPU information.

        Returns:
            Dictionary containing CPU specifications and current usage
        """
        cpu_freq = psutil.cpu_freq()
        return {
            'cpu_count_physical': psutil.cpu_count(logical=False),  # Physical cores
            'cpu_count_logical': psutil.cpu_count(logical=True),     # With hyperthreading
            'cpu_percent': psutil.cpu_percent(interval=1),           # Current usage %
            'cpu_freq_current_mhz': cpu_freq.current if cpu_freq else None,
            'cpu_freq_max_mhz': cpu_freq.max if cpu_freq else None,
            'processor': platform.processor(),                        # CPU model name
            'architecture': platform.machine(),                       # x86_64, arm64, etc.
        }

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get RAM information (current usage at checkpoint save time).

        Returns:
            Dictionary containing RAM total, used, and percentage
        """
        mem = psutil.virtual_memory()
        return {
            'ram_total_gb': round(mem.total / (1024**3), 2),
            'ram_current_used_gb': round(mem.used / (1024**3), 2),
            'ram_current_percent': mem.percent,
        }

    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        Get GPU VRAM usage - both current and peak.

        Current usage: Memory allocated at checkpoint save time (usually low)
        Peak usage: Maximum memory used since torch.cuda.reset_peak_memory_stats()

        Returns:
            Dictionary with GPU memory info for all available devices
        """
        if not torch.cuda.is_available():
            return {'gpu_available': False}

        devices = []
        for i in range(torch.cuda.device_count()):
            devices.append({
                'device_id': i,
                'name': torch.cuda.get_device_name(i),
                'total_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),

                # Current usage (at checkpoint save time - usually low)
                'current_allocated_gb': round(torch.cuda.memory_allocated(i) / (1024**3), 2),
                'current_reserved_gb': round(torch.cuda.memory_reserved(i) / (1024**3), 2),

                # PEAK usage (maximum during training - CRITICAL for reproducibility!)
                'peak_allocated_gb': round(torch.cuda.max_memory_allocated(i) / (1024**3), 2),
                'peak_reserved_gb': round(torch.cuda.max_memory_reserved(i) / (1024**3), 2),
            })

        return {
            'gpu_available': True,
            'gpu_count': torch.cuda.device_count(),
            'devices': devices,
        }

    def get_device_info(self, device) -> Dict[str, Any]:
        info = {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": device,
        }
        if device=="cuda":
            dev = torch.cuda.get_device_properties(0)
            info.update(
                {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_total_memory_bytes": dev.total_memory,
                }
            )
        return info

    def save_ckpt(
        self,
        epoch: int,
        step: int,
        train_loss_history: List[float],
        val_loss_history: List[float],
        best_val_loss: float,
        patience_counter: int,
        model_save: bool = False
    ) -> None:
        '''
        Save model checkpoint with training state and reproducibility info.

        Saves model weights, optimizer state, loss history, random states, environment info,
        and comprehensive system information including CPU, RAM, and GPU memory usage.
        Note: Model weights are large - consider saving less frequently.

        Args:
            epoch (int): Current training epoch number (0-indexed or 1-indexed based on your convention).
            step (int): Current training step/iteration number (total number of batches processed).
            train_loss_history (List[float]): List of training losses for each epoch.
                Example: [2.5, 2.3, 2.1, 1.9]
            val_loss_history (List[float]): List of validation losses for each epoch.
                Example: [2.6, 2.4, 2.2, 2.0]
            best_val_loss (float): Best validation loss achieved so far during training.
                Example: 2.0
            patience_counter (int): Current patience counter value for early stopping.
                Increments when validation loss doesn't improve.
                Example: 0 (just improved), 3 (3 epochs without improvement)
            model_save (bool, optional): Whether to save model weights, optimizer state, and scheduler state.
                Set to True for periodic checkpoints or best model saves.
                Set to False for lightweight logging checkpoints (saves ~90% disk space).
                Defaults to False.
        '''

        training_time = time.time() - self.time_settings.start_time
        self.result_log = {
            'epoch': epoch,
            'step': step,
            'epoch_loss': {
                'train_loss': train_loss_history,   # list[float]: [{epo: train_loss}, ...]
                'val_loss':   val_loss_history,     # list[float]: [{epo: val_loss}, ...]
                'best_val_loss': best_val_loss,     # {epo: best_val_loss}
                'patience_counter': patience_counter, # {epo: patience_counter}
            },
            'seed': {
                "seed": self.seed_settings.seed,
                "random_state": random.getstate(),
                "numpy_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
                "cuda_state": torch.cuda.get_rng_state_all()
            },
            'git': {
                'commit_hash': self.git_settings.git_commit_hash,
                'branch': self.git_settings.git_branch,
                'is_dirty': self.git_settings.uncommit_git_exist,
            },
            'config': self.deep_learning_settings.config,
            'cli_args': sys.argv,  # Command-line arguments for reproducibility
            'run_dir': self.path_settings.file_path,
            'training_time': training_time,
            'executed_datetime': {
                'start_datetime': self.time_settings.start_datetime,
                'end_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            'env': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'os': os.name,
                'python_version': sys.version,
            },
            'system_info': {
                'cpu': self.get_cpu_info(),
                'gpu': self.get_device_info(self.env_settings.device),
                'memory': {
                    **self.get_memory_info()  # Peak RAM during training
                },
                'gpu_memory': self.get_gpu_memory_info(),  # Includes both current and peak VRAM
            },
            'user_notes': self.path_settings.user_note,
        }
        if model_save:
            self.result_log.update({
                'model_state_dict': self.deep_learning_settings.model.state_dict(),
                'optimizer_state_dict': self.deep_learning_settings.optimizer.state_dict(),
                'scheduler_state_dict': self.deep_learning_settings.scheduler.state_dict() if self.deep_learning_settings.scheduler else None
            })
        torch.save(self.result_log, Path(self.path_settings.save_path) / f"{self.path_settings.save_name}.pt")