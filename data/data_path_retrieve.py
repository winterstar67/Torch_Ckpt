"""
ipynb에서 사용하는 데이터셋 경로를 반환하는 함수

참고: nanoGPT/data/shakespeare/prepare.py


예시:
- data_path_retrieve('shakespeare')
- data_path_retrieve('openwebtext')
- data_path_retrieve('shakespeare_char')
"""

import os

def data_path_retrieve():
    return os.path.join(os.path.dirname(__file__))