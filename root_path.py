from pathlib import Path

def get_root_path() -> str:
    return Path(__file__).resolve().parent