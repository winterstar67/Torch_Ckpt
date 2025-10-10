'''
Settings classes for checkpoint manager.

This module contains all the configuration classes used by ckpt_manager:
- SeedSettings: Random seed configuration
- DLSettings: Deep learning components (model, optimizer, scheduler)
- PathSettings: File paths and user notes
- GitSettings: Git tracking configuration
- EnvSettings: Environment configuration
- TimeSettings: Timestamp tracking
'''

import os
import time
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any
import inspect
import platform

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class SettingsBase:
    required_fields: dict = {}   # key → expected type
    KEY_HELP: dict = {}
    UNSET=object()

    def validate(self, settings: dict):
        # 1. Missing keys
        self.missing = [key for key, value in settings.items() if self.is_UNSET(value)]


        if self.missing:
            raise ValueError(self._build_missing_error(self.missing))

        # 2. Type mismatches
        type_errors = []
        for key, expected_type in self.required_fields.items():
            if not isinstance(settings[key], expected_type):
                type_errors.append((key, expected_type, type(settings[key])))

        if type_errors:
            raise ValueError(self._build_type_error(type_errors))

    def _get_error_location(self, levels_back=0):
        """
        Get formatted error location string.

        Args:
            levels_back (int): How many frames to go back from the caller.
                              0 = caller's location (default)
                              1 = caller's caller, etc.

        Returns:
            str: Formatted location string like "[File: settings.py | Class: SeedSettings | Function: __init__]"
        """
        file_name = os.path.basename(__file__)
        class_name = self.__class__.__name__

        # Get the caller's frame
        frame = inspect.currentframe().f_back
        for _ in range(levels_back):
            if frame and frame.f_back:
                frame = frame.f_back

        function_name = frame.f_code.co_name if frame else "<unknown>"
        return f"[File: {file_name} | Class: {class_name} | Function: {function_name}]"

    def _build_missing_error(self, missing):
        location = self._get_error_location(levels_back=1)
        lines = [f"{location}\nMissing keys in {missing}:\n"]
        for key in missing:
            help_msg = self.KEY_HELP.get(key, f"'{key}' is missing.")
            lines.append(help_msg + "\n")
        return "\n".join(lines)

    def _build_type_error(self, errors):
        location = self._get_error_location(levels_back=1)
        lines = [f"{location}\nType errors in {errors}:\n"]
        for key, expected, actual in errors:
            lines.append(
                f"'{key}' has wrong type.\n"
                f"  → Expected: {expected}\n"
                f"  → Got:      {actual}\n"
                f"{self.KEY_HELP.get(key, '')}\n"
            )
        return "\n".join(lines)

    def is_UNSET(self, x):
        return x == self.UNSET



class SeedSettings(SettingsBase):
    """All seed-related knobs."""
    required_fields = {
            "use_seed":bool,
            "seed":int,
            "use_deterministic":bool
        }

    KEY_HELP = {
        "use_seed": (
            "'use_seed' is missing.\n"
            "  → Meaning: Whether to apply seed settings.\n"
            "  → Example: use_seed=True"
        ),
        "seed": (
            "'seed' is missing.\n"
            "  → Meaning: The base integer seed.\n"
            "  → Example: seed=42"
        ),
        "use_deterministic": (
            "'use_deterministic' is missing.\n"
            "  → Meaning: Use PyTorch deterministic mode.\n"
            "  → Example: use_deterministic=False"
        ),
    }

    def __init__(self, settings: dict):
        """
        Initialize seed settings.

        Args:
            use_seed (bool): Whether to actually apply the seed settings
            seed (int): Base random seed for all RNGs. Default: 42
            use_deterministic (bool): Use torch deterministic algorithms (slower but reproducible). Default: False
        """
        self.use_seed = settings.get('use_seed',self.UNSET)
        self.seed = settings.get('seed',self.UNSET)
        self.use_deterministic = settings.get('use_deterministic',self.UNSET)
        if self.use_deterministic == True:
            print("use_deterministic == True")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        self.validate(self.__dict__)


class DLSettings(SettingsBase):
    """Deep learning config / hyper-parameters."""
    required_fields = {
        "data_config": dict,
        "model_config": dict,
        "optimizer_config": dict,
        "trainer_config": dict
    }

    KEY_HELP = {
        "data_config": (
            "'data_config' is missing.\n"
            "  → Meaning: Data configuration dictionary.\n"
            "  → Example: data_config={'batch_size': 32, 'train_path': 'data/train.bin', 'val_path': 'data/val.bin'}"
        ),
        "model_config": (
            "'model_config' is missing.\n"
            "  → Meaning: Model configuration dictionary.\n"
            "  → Example: model_config={'encoder_name': 'gpt2', 'embed_dim': 768, 'token_len': 1024, 'n_head': 6, 'ff_dim': 2048, 'n_blocks': 4}"
        ),
        "optimizer_config": (
            "'optimizer_config' is missing.\n"
            "  → Meaning: Optimizer configuration dictionary.\n"
            "  → Example: optimizer_config={'lr': 0.001, 'optimizer': 'Adam'}"
        ),
        "trainer_config": (
            "'trainer_config' is missing.\n"
            "  → Meaning: Trainer configuration dictionary.\n"
            "  → Example: trainer_config={'max_epochs': 100, 'patience': 10, 'min_delta': 0.001, 'verbose': 1, 'save_model': True, 'save_model_path': 'models/model.pth', 'save_model_name': 'model.pth'}"
        )
    }

    def __init__(self, settings: dict):
        """
        Initialize deep learning settings.

        Args:
            settings (dict): Dictionary containing:
                - data_config (dict): Data configuration dictionary.
                - model_config (dict): Model configuration dictionary.
                - optimizer_config (dict): Optimizer configuration dictionary.
                - trainer_config (dict): Trainer configuration dictionary.
        """
        self.data_config = settings.get('data_config', self.UNSET)
        self.model_config = settings.get('model_config', self.UNSET)
        self.optimizer_config = settings.get('optimizer_config', self.UNSET)
        self.trainer_config = settings.get('trainer_config', self.UNSET)
        # These will be set later by the user
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None

        self.validate(self.__dict__)


class PathSettings(SettingsBase):
    """Paths for runs and checkpoints."""
    required_fields = {
        "file_path": str,
        "save_path": str,
        "save_name": str,
        "user_note": str
    }

    KEY_HELP = {
        "file_path": (
            "'file_path' is missing.\n"
            "  → Meaning: Directory for logs and run artifacts.\n"
            "  → Example: file_path='./logs'"
            "  **If you run the jupyter file, you must input the file_path manually"
        ),
        "save_path": (
            "'save_path' is missing.\n"
            "  → Meaning: Directory to save checkpoints.\n"
            "  → Example: save_path='./checkpoints'"
        ),
        "save_name": (
            "'save_name' is missing.\n"
            "  → Meaning: Name of the checkpoint file.\n"
            "  → Example: save_name='model.pth'"
        ),
        "user_note": (
            "'user_note' is missing.\n"
            "  → Meaning: Free-form notes about this run.\n"
            "  → Example: user_note='Testing new architecture'"
        )
    }

    def __init__(self, settings: dict):
        """
        Initialize path settings.

        Args:
            settings (dict): Dictionary containing:
                - file_path (str): Directory for logs / run artifacts
                - save_path (str): Directory to save checkpoints
                - save_name (str): Name of the checkpoint file. Default: ""
                - user_note (str): Free-form notes about this run. Default: ""
        """
        if settings.get('file_path', self.UNSET) != self.UNSET:
            self.file_path = settings.get('file_path', self.UNSET)
        else:
            self.file_path = self._get_caller_file_dir()
        self.save_path = settings.get('save_path', self.UNSET)
        self.save_name = settings.get('save_name', self.UNSET)
        self.user_note = settings.get('user_note', "")

        self.validate(self.__dict__)

    def _get_caller_file_dir(self):
        stack = inspect.stack()
        for frame_info in stack:
            if frame_info.filename.startswith('<ipython-input') or 'ipykernel' in frame_info.filename:
                raise ValueError("""
                You're in jupyter notebook, must input the path of ipynb file yourself.
                Example: 
                    with open("../config.json", "r") as f:
                        config = json.load(f)
                    config['path_settings']['file_path'] = "{file_directory}/ipynb_file_name.ipynb"
                """)
                
            if ("settings.py" not in frame_info.filename) and ("torch_ckpt.py" not in frame_info.filename):
                return os.path.dirname(os.path.abspath(frame_info.filename))
        
        return "Please make sure that your file name is not settings.py or torch_ckpt.py"

class GitSettings(SettingsBase):
    """Git + environment logging options."""
    required_fields = {
        "use_git": bool,
        "strict_git": bool
    }

    KEY_HELP = {
        "use_git": (
            "'use_git' is missing.\n"
            "  → Meaning: Enable git tracking for this run.\n"
            "  → Example: use_git=True"
        ),
        "strict_git": (
            "'strict_git' is missing.\n"
            "  → Meaning: Raise error if not a clean git repo instead of warning.\n"
            "  → Example: strict_git=False"
        )
    }

    def __init__(self, settings: dict):
        """
        Initialize git settings.

        Args:
            settings (dict): Dictionary containing:
                - use_git (bool): Enable git tracking for this run. Default: True
                - strict_git (bool): Raise error if not a clean git repo instead of warning. Default: False
        """
        self.use_git = settings.get('use_git', self.UNSET)
        self.strict_git = settings.get('strict_git', self.UNSET)

        # Fields initialized based on git state
        self.git_commit_hash: Optional[str] = None
        self.git_branch: Optional[str] = None
        self.git_is_dirty: Optional[bool] = None

        self.validate(self.__dict__)

        # Get git info if enabled
        if self.use_git:
            self.get_git_info()
        else:
            self.git_commit_hash, self.git_branch, self.git_is_dirty = [None] * 3

    def get_git_info(self):
        """
        Get git information: commit hash, branch name, and dirty state.

        Behavior modes:
        - use_git=False: Skip git checks entirely
        - strict_git=False (default): Print warnings, set values to None on errors
        - strict_git=True: Raise exceptions on errors
        """

        # Step 1: Check if git repository exists
        try:
            subprocess.run(['git', 'rev-parse', '--git-dir'],
                          capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            msg = "[Git] Not a git repository. Run 'git init' to enable git tracking."
            if self.strict_git:
                raise RuntimeError(msg)
            else:
                print(f"[Warning] {msg}")
                self.git_commit_hash = None
                self.git_branch = None
                self.git_is_dirty = None
                return  # Early return

        # Step 2: Get branch name
        try:
            branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                    capture_output=True, text=True, check=True)
            self.git_branch = branch.stdout.strip()
        except subprocess.CalledProcessError:
            msg = "[Git] Could not get branch name."
            if self.strict_git:
                raise RuntimeError(msg)
            else:
                print(f"[Warning] {msg}")
                self.git_branch = None

        # Step 3: Get commit hash
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                    capture_output=True, text=True, check=True)
            self.git_commit_hash = result.stdout.strip()
        except subprocess.CalledProcessError:
            msg = "[Git] No commits yet. Make your first commit for reproducibility."
            if self.strict_git:
                raise RuntimeError(msg)
            else:
                print(f"[Warning] {msg}")
                self.git_commit_hash = None
                self.git_is_dirty = None
                return  # Can't check dirty state without commits

        # Step 4: Check for uncommitted changes (dirty state)
        result = subprocess.run(['git', 'diff-index', '--quiet', 'HEAD'],
                               capture_output=True)
        self.git_is_dirty = (result.returncode != 0)

        if self.git_is_dirty:
            msg = "[Git] Uncommitted changes detected! Commit them for full reproducibility."
            if self.strict_git:
                raise RuntimeError(f"⚠️  {msg}")
            else:
                print(f"⚠️  Warning: {msg}")


class EnvSettings(SettingsBase):
    """Environment configuration settings."""
    required_fields = {
        "device": str,
        "save_code": bool,
        "save_requirements_txt": bool
    }

    KEY_HELP = {
        "device": (
            "'device' is missing.\n"
            "  → Meaning: Device to use for computation.\n"
            "  → Example: device='cuda' or device='cpu'"
        ),
        "save_code": (
            "'save_code' is missing.\n"
            "  → Meaning: Save code file snapshot at the start of the run.\n"
            "  → Example: save_code=True"
        ),
        "save_requirements_txt": (
            "'save_requirements_txt' is missing.\n"
            "  → Meaning: Save requirements.txt snapshot at the start of the run.\n"
            "  → Example: save_requirements_txt=True"
        )
    }

    def __init__(self, settings: dict):
        """
        Initialize environment settings.

        Args:
            settings (dict): Dictionary containing:
                - device (str): Device to use: 'cuda' or 'cpu'
                - save_code (bool): Save code file snapshot at the start of the run
                - save_requirements_txt (bool): Save requirements.txt snapshot at the start of the run
        """
        self.device = settings.get('device', self.UNSET)
        self.save_code = settings.get('save_code', self.UNSET)
        self.save_requirements_txt = settings.get('save_requirements_txt', self.UNSET)

        self.validate(self.__dict__)


class TimeSettings(SettingsBase):
    """Timestamp tracking settings."""
    required_fields = {
        "start_time": float,
        "start_datetime": str,
        "start_date": str
    }

    KEY_HELP = {
        "start_time": (
            "'start_time' is missing.\n"
            "  → Meaning: Unix timestamp when the run started.\n"
            "  → This is auto-generated from time.time()"
        ),
        "start_datetime": (
            "'start_datetime' is missing.\n"
            "  → Meaning: Human-readable datetime when the run started.\n"
            "  → This is auto-generated from datetime.now()"
        ),
        "start_date": (
            "'start_date' is missing.\n"
            "  → Meaning: Formatted date string for the run.\n"
            "  → This is auto-generated from datetime.now()"
        )
    }

    def __init__(self):
        """
        Initialize time settings with current timestamp.

        Args:
            settings (dict): Dictionary (can be empty as times are auto-generated)
        """
        self.start_time = time.time()
        self.start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.start_date = datetime.now().strftime("%Y-%m-%d-%H%M")

        self.validate(self.__dict__)
