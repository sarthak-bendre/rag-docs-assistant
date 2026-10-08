"""Central settings, read from environment variables / .env."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    pdf_dir: Path = Path(os.getenv("PDF_DIR", ROOT / "data" / "pdfs"))
    index_dir: Path = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
    collection: str = "manuals"

    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", 500))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 80))

    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    # bge models expect this prefix on queries (not on passages)
    query_prefix: str = os.getenv(
        "QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
    )
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Query-time models are small; keep them on CPU so the GPU stays free for the Ollama LLM
    query_device: str = os.getenv("QUERY_DEVICE", "cpu")

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    judge_provider: str = os.getenv("JUDGE_PROVIDER", os.getenv("LLM_PROVIDER", "ollama"))
    judge_model: str = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "llama3.2:3b"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"


settings = Settings()
