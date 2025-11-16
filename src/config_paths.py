# src/config_paths.py

from pathlib import Path

# directorio base del proyecto (src -> parent)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SRC_DIR = BASE_DIR / "src"
