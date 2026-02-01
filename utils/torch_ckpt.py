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


# === Path Finding Utilities ===
def find_root(proj_name: str) -> str:
    """
    Find the project root directory by searching upward from this file's location.

    Args:
        proj_name: Name of the project directory to find

    Returns:
        Absolute path to the project root directory

    Raises:
        FileNotFoundError: If project directory is not found within 100 parent levels
    """
    dir_path = __file__
    for _ in range(100):
        dir_path = os.path.dirname(dir_path)
        # Handle both Windows and Linux path separators
        last_dir = os.path.basename(dir_path)
        if last_dir == proj_name:
            return dir_path
    raise FileNotFoundError(f"Project directory '{proj_name}' not found")


def find_from_proj(proj_name: str, target: str) -> str:
    """
    Find a file or directory starting from the project root using BFS.

    Args:
        proj_name: Name of the project directory
        target: Name of the file or directory to find

    Returns:
        Absolute path to the target

    Raises:
        FileNotFoundError: If target is not found
    """
    proj_dir = find_root(proj_name)

    # BFS to find target
    queue = [proj_dir]
    while queue:
        current_dir = queue.pop(0)
        try:
            entries = os.listdir(current_dir)
        except PermissionError:
            continue

        if target in entries:
            return os.path.join(current_dir, target)

        # Add subdirectories to queue (skip hidden directories)
        for entry in entries:
            entry_path = os.path.join(current_dir, entry)
            if os.path.isdir(entry_path) and not entry.startswith('.'):
                queue.append(entry_path)

    raise FileNotFoundError(f"Target '{target}' not found in project '{proj_name}'")


class ckpt_manager:
    """Checkpoint manager for tracking training state and reproducibility."""

    def __init__(
        self,
        proj_dir:str,
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
        self.proj_dir = proj_dir
        self.seed_settings = SeedSettings(seed_settings)
        self.deep_learning_settings = DLSettings(deep_learning_settings)
        self.path_settings = PathSettings(path_settings)
        self.git_settings = GitSettings(git_settings, proj_dir=proj_dir)
        self.env_settings = EnvSettings(env_settings)

        # Initialize time settings
        self.time_settings = TimeSettings()

        # Create unified session directory: {proj_dir}/{save_path}/ckpt_result_{timestamp}/
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.path_settings.save_path:
            self.session_dir = Path(self.proj_dir) / self.path_settings.save_path / f"ckpt_result_{self.session_timestamp}"
        else:
            self.session_dir = Path(self.proj_dir) / "ckpt" / f"ckpt_result_{self.session_timestamp}"
        os.makedirs(str(self.session_dir), exist_ok=True)

        # Apply seed settings
        self.set_seed(self.seed_settings.seed)
        if self.seed_settings.use_deterministic:
            torch.use_deterministic_algorithms(True)

        # Save requirements if requested (in session directory)
        if self.env_settings.save_requirements_txt:
            self.save_requirements(str(self.session_dir))

        # Reset GPU memory stats to track peak from training start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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

    def git_ignore_pattern_extraction(self, proj_dir: str):
        with open(os.path.join(proj_dir, ".gitignore"), "r") as f:
            ignore_patterns = [i.replace("\n","") for i in x if not((i.startswith("#")) or ((i=='\n')) or (i=="")) ]
        return ignore_patterns

    def get_all_project_files(self, proj_dir: Path) -> List[str]:
        """
        Get all files in project directory, excluding hidden files/folders (starting with '.').

        Args:
            proj_dir: Project root directory

        Returns:
            List of relative file paths
        """
        all_files = []
        for root, dirs, files in os.walk(proj_dir):
            # Filter out hidden directories (modifies dirs in-place to skip them)
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for file in files:
                if not file.startswith('.'):
                    rel_path = Path(root).relative_to(proj_dir) / file
                    # Convert to string and handle root-level files
                    rel_str = str(rel_path)
                    if rel_str != '.':
                        all_files.append(rel_str)
        return all_files

    def generate_directory_tree(
        self,
        file_list: List[str],
        root_name: str = ".",
        max_files_per_dir: int = 5,
        backed_up_files: Optional[List[str]] = None
    ) -> str:
        """
        Generate a directory tree string from a list of file paths.

        Args:
            file_list: List of relative file paths to display
            root_name: Name to display as root directory
            max_files_per_dir: Maximum number of files to display per directory (default: 5)
            backed_up_files: List of backed-up file paths. If provided, marks non-backed-up items.

        Returns:
            String representation of directory tree with ASCII characters
        """
        # Build tree structure as nested dict
        tree = {}
        for file_path in file_list:
            parts = Path(file_path).parts
            current = tree
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        # Build set of backed-up files and directories for quick lookup
        backed_up_set = set(backed_up_files) if backed_up_files else None
        backed_up_dirs = set()
        if backed_up_files:
            for f in backed_up_files:
                # Add all parent directories of backed-up files
                parts = Path(f).parts
                for i in range(len(parts)):
                    backed_up_dirs.add("/".join(parts[:i+1]))

        # Generate tree string
        lines = [root_name]

        def _build_tree(node: dict, prefix: str = "", current_path: str = "", parent_backed_up: bool = True):
            # Separate directories (non-empty dict) and files (empty dict)
            dirs = sorted([name for name, children in node.items() if children])
            files = sorted([name for name, children in node.items() if not children])

            # Combine: directories first, then files
            num_files = len(files)

            # Calculate if we need to truncate files
            truncate_files = num_files > max_files_per_dir
            if truncate_files:
                files_to_show = files[:max_files_per_dir]
                display_items = dirs + files_to_show + ["..."]
            else:
                display_items = dirs + files

            for i, name in enumerate(display_items):
                is_last = (i == len(display_items) - 1)
                connector = "└── " if is_last else "├── "

                if name == "...":
                    remaining = num_files - max_files_per_dir
                    lines.append(f"{prefix}{connector}... ({remaining} more files)")
                else:
                    item_path = f"{current_path}/{name}" if current_path else name

                    # Determine if this item is backed up
                    is_dir = name in dirs
                    if is_dir:
                        is_backed_up = item_path in backed_up_dirs or item_path.replace("/", "\\") in backed_up_dirs
                    else:
                        is_backed_up = backed_up_set is None or item_path in backed_up_set or item_path.replace("/", "\\") in backed_up_set

                    # Add "(not backuped)" marker only if parent is backed up
                    suffix = ""
                    if backed_up_set is not None and not is_backed_up and parent_backed_up:
                        suffix = " (not backuped)"

                    lines.append(f"{prefix}{connector}{name}{suffix}")

                    if is_dir:
                        extension = "    " if is_last else "│   "
                        _build_tree(node[name], prefix + extension, item_path, parent_backed_up=is_backed_up)

        _build_tree(tree)
        return "\n".join(lines)

    def save_directory_tree(
        self,
        proj_dir: Path,
        backed_up_files: List[str],
        save_path: Path,
        root_name: str = "."
    ) -> None:
        """
        Save directory tree to a .txt file.

        Args:
            proj_dir: Project root directory
            backed_up_files: List of backed-up file paths
            save_path: Path to save the tree file
            root_name: Name to display as root directory
        """
        # Get all project files (excluding hidden files/folders)
        all_files = self.get_all_project_files(proj_dir)
        total_files = len(all_files)
        backed_up_count = len(backed_up_files)

        tree_str = self.generate_directory_tree(all_files, root_name, backed_up_files=backed_up_files)
        tree_file_path = save_path / "directory_tree.txt"

        with open(str(tree_file_path), "w", encoding="utf-8") as f:
            f.write(f"Directory Tree - Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(tree_str)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write(f"Total files in project: {total_files}\n")
            f.write(f"Backed up files: {backed_up_count}\n")

    def backup_files(
        self,
        proj_dir: Path,
        git_track: bool = True,
        custom_file_paths: Optional[List[str]] = None
    ):
        # Use unified session directory: {save_path}/ckpt_result_{timestamp}/ckpt_backup/
        dst_dir = self.session_dir / "ckpt_backup"
        os.makedirs(str(dst_dir), exist_ok=True)

        # Validate custom_file_paths
        has_custom_files = custom_file_paths is not None and custom_file_paths != "" and len(custom_file_paths) > 0

        if not git_track and not has_custom_files:
            raise ValueError("custom_file_paths must be provided when git_track=False")

        if git_track and not has_custom_files:
            print("[Warning] No custom_file_paths provided. Only git-tracked files will be backed up.")

        # Validate that custom files exist
        if has_custom_files:
            for file_path in custom_file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Custom file not found: {file_path}")

        # Build file list based on git_track and custom_file_paths
        file_list = []

        if git_track:
            cmd = ["git", "-C", str(proj_dir), "ls-files", "-c", "-o", "--exclude-standard"]
            git_file_list = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()
            file_list.extend(git_file_list)

        if has_custom_files:
            # Convert custom paths to relative paths from proj_dir
            for file_path in custom_file_paths:
                abs_path = Path(file_path).resolve()
                try:
                    rel_path = abs_path.relative_to(proj_dir.resolve())
                    if str(rel_path) not in file_list:
                        file_list.append(str(rel_path))
                except ValueError:
                    # File is outside proj_dir, copy to root of backup
                    file_name = abs_path.name
                    if file_name not in file_list:
                        file_list.append(f"__external__/{file_name}")
                        # Copy external file directly
                        external_dir = dst_dir / "__external__"
                        os.makedirs(str(external_dir), exist_ok=True)
                        shutil.copy2(str(abs_path), str(external_dir / file_name))

        path_pair = []
        backup_success = True

        def dt(ts: float) -> str:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        # Phase 1: Copy all files creating the directories
            # The mtime, atime of the directories would be the same as the source files with path_pair in Phase 2
        for name in file_list:
            # Skip external files (already copied above)
            if name.startswith("__external__/"):
                continue
            try:
                path_list = Path(name).parents # []
                for each_path in reversed(list(path_list)): # reverse 결과 순서 [PosixPath('.'), PosixPath('utils'), PosixPath('utils/ckpt_modules')]
                    if each_path == Path("."):
                        continue
                    src_path = proj_dir/each_path
                    dst_path = dst_dir/each_path
                    os.makedirs(str(dst_path), exist_ok=True)
                    # Always add to path_pair to update directory stats on every backup
                    if src_path != proj_dir and [src_path, dst_path] not in path_pair:
                        path_pair.append([src_path, dst_path])
                shutil.copy2(proj_dir/name, dst_dir/name)
            except Exception as e:
                with open(str(dst_dir/'error_log.txt'), 'a') as f:
                    f.write(f"Error copying file {name}: {e}\n\n")
                backup_success = False

        path_pair.sort(key=lambda x: len(str(x[1])), reverse=True)

        # Phase 2: Copy the directory metadata
        for src_path, dst_path in path_pair:
            shutil.copystat(str(src_path), str(dst_path))

        # Phase 3: Check whether the metadata is correct
        for name in file_list:
            # Skip external files
            if name.startswith("__external__/"):
                continue
            path_list = list(Path(name).parents)[:-1] # 마지막의 .는 제외한다. - ## Need more absolute and clear logic
            all_src_path = list(map(lambda x: proj_dir/x, path_list))
            all_dst_path = list(map(lambda x: dst_dir/x, path_list))
            all_pairs = list(zip(all_src_path, all_dst_path))
            # I'm not sure whether the access time can be same when we check it because I can access to two files at the same time
            # if not all(map(lambda x: dt(x[0].stat().st_atime) == dt(x[1].stat().st_atime), all_pairs)):
            #     with open(str(dst_dir/'error_log.txt'), 'a') as f:
            #         f.write(f"Error in copying access time {name}\n\n")
            #     print(f"{name}'s mtime is not same with original data")

            if not all(map(lambda x: dt(x[0].stat().st_mtime) == dt(x[1].stat().st_mtime), all_pairs)):
                with open(str(dst_dir/'error_log.txt'), 'a') as f:
                    f.write(f"Error in copying modified time {name}\n\n")
                print(f"{name}'s mtime is not same with original data")
                backup_success = False
                # Failed to backup

        # Save directory tree to .txt file (shows full project, marks non-backed-up items)
        self.save_directory_tree(proj_dir, file_list, dst_dir, root_name=proj_dir.name)

        return backup_success

    def _get_current_lr(self) -> Optional[float]:
        """
        Get current learning rate from optimizer.

        Returns:
            Current learning rate, or None if optimizer is not set.
            If multiple param groups exist, returns the first group's lr.
        """
        if self.deep_learning_settings.optimizer is None:
            return None
        # Get lr from first param group (most common case)
        return self.deep_learning_settings.optimizer.param_groups[0]['lr']

    def save_ckpt(
        self,
        step: int,
        train_loss_history: Dict[int, Dict[str, float]],
        val_loss_history: Dict[int, Dict[str, float]],
        best_val_loss: float,
        patience_counter: int,
    ) -> None:
        '''
        Save model checkpoint with training state and reproducibility info.

        Saves model weights, optimizer state, loss history, random states, environment info,
        and comprehensive system information including CPU, RAM, and GPU memory usage.
        Note: Model weights are large - consider saving less frequently.

        Config options (env_settings in config.json):
            - model_save: Whether to save model weights, optimizer state, and scheduler state
            - save_code: Whether to backup the project worktree files and save the directory tree.
                When enabled, both file backup and directory tree generation are performed together
                (these two operations are not separable).

        Args:
            step (int): Current training step/iteration number (total number of batches processed).
            train_loss_history (Dict[int, Dict[str, float]]): Dict mapping iteration to loss info.
                Each entry contains {"loss": float, "batch_size": int}.
                Example: {0: {"loss": 2.5, "batch_size": 32}, 1: {"loss": 2.4, "batch_size": 32}, ...}
            val_loss_history (Dict[int, Dict[str, float]]): Dict mapping iteration to loss info.
                Each entry contains {"loss": float, "batch_size": int}.
                Example: {0: {"loss": 2.6, "batch_size": 32}, 1: {"loss": 2.5, "batch_size": 32}, ...}
            best_val_loss (float): Best validation loss achieved so far during training.
                Example: 2.0
            patience_counter (int): Current patience counter value for early stopping.
                Increments when validation loss doesn't improve.
                Example: 0 (just improved), 3 (3 epochs without improvement)
        '''

        training_time = time.time() - self.time_settings.start_time
        self.result_log = {
            'step': step,
            'loss_history': {
                'train_loss': train_loss_history,   # Dict[int, Dict]: {iter: {"loss": float, "batch_size": int}, ...}
                'val_loss':   val_loss_history,     # Dict[int, Dict]: {iter: {"loss": float, "batch_size": int}, ...}
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
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
                'is_dirty': self.git_settings.git_is_dirty,
            },
            'config': {
                'data_config': self.deep_learning_settings.data_config,
                'model_config': self.deep_learning_settings.model_config,
                'optimizer_config': self.deep_learning_settings.optimizer_config,
                'trainer_config': self.deep_learning_settings.trainer_config,
            },
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
            'learning_rate': self._get_current_lr(),
        }
        if self.env_settings.model_save:
            self.result_log.update({
                'model_state_dict': self.deep_learning_settings.model.state_dict(),
                'optimizer_state_dict': self.deep_learning_settings.optimizer.state_dict(),
                'scheduler_state_dict': self.deep_learning_settings.scheduler.state_dict() if self.deep_learning_settings.scheduler else None,
                'grad_scaler_state_dict': self.deep_learning_settings.grad_scaler.state_dict() if self.deep_learning_settings.grad_scaler else None
            })

        # Save checkpoint in unified session directory: {save_path}/ckpt_result_{timestamp}/{save_name}.pt
        save_path = self.session_dir / f"{self.path_settings.save_name}.pt"
        torch.save(self.result_log, save_path)

        backup_success = True
        if self.env_settings.save_code:
            backup_success = self.backup_files(proj_dir=Path(self.proj_dir))

        # Print completion status
        if backup_success:
            print(f"[Checkpoint] Saved successfully to: {self.session_dir}")
        else:
            print(f"[Checkpoint] Saved to: {self.session_dir} (Warning: some backup files failed - check error_log.txt)")
