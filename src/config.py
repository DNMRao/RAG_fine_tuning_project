"""Read .env once so every module uses the same settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # copies KEY=value lines from .env into os.environ

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
EMBEDDING_REVISION = os.getenv("EMBEDDING_REVISION") or None
if EMBEDDING_REVISION is None:
    if EMBEDDING_MODEL != DEFAULT_EMBEDDING_MODEL:
        raise ValueError("Set EMBEDDING_REVISION when changing EMBEDDING_MODEL")
    EMBEDDING_REVISION = DEFAULT_EMBEDDING_REVISION

TOP_K = int(os.getenv("TOP_K") or 3)
MIN_SCORE = float(os.getenv("MIN_SCORE") or 0.30)
VECTOR_STORAGE_PATH = os.getenv("VECTOR_STORAGE_PATH") or None  # None = in-memory

DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_BASE_MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"

BASE_MODEL = os.getenv("BASE_MODEL") or DEFAULT_BASE_MODEL
BASE_MODEL_REVISION = os.getenv("BASE_MODEL_REVISION") or None
if BASE_MODEL_REVISION is None:
    if BASE_MODEL != DEFAULT_BASE_MODEL:
        raise ValueError("Set BASE_MODEL_REVISION when changing BASE_MODEL")
    BASE_MODEL_REVISION = DEFAULT_BASE_MODEL_REVISION

ADAPTER_PATH = Path(os.getenv("ADAPTER_PATH") or "artifacts/support-lora-adapter")
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH") or 256)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS") or 80)
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS") or 8)
LEARNING_RATE = float(os.getenv("LEARNING_RATE") or 2e-4)
RANDOM_SEED = int(os.getenv("RANDOM_SEED") or 42)
