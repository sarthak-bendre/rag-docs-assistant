# Load .env before any submodule imports huggingface/sentence-transformers (HF_HOME must be set first)
from dotenv import load_dotenv as _load
from pathlib import Path as _P

_load(_P(__file__).resolve().parent.parent / ".env")
