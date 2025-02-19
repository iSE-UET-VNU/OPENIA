import getpass
import os
import sys

__USERNAME = getpass.getuser()
_BASE_DIR = f''
MODEL_PATH = f'{_BASE_DIR}/weights/'
BENCHMARK_FOLDER = os.path.join(_BASE_DIR, 'benchmark')
DATA_FOLDER = os.path.join(_BASE_DIR, 'datasets')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
