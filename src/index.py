"""Step 2 - Index: embed chunks and store them in Chroma (+ a JSONL copy for BM25).

Run:  python -m src.index
"""
import json
import shutil

import chromadb
from sentence_transformers import SentenceTransformer

from .config import settings
from .ingest import chunk_folder


def build_index(batch_size: int = 64) -> None:
    print("Chunking PDFs...")
    chunks = chunk_folder()
    print(f"Total chunks: {len(chunks)}")

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    print(f"Embedding with {settings.embed_model}...")
    model = SentenceTransformer(settings.embed_model)
    vectors = model.encode(
        [c.text for c in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    col = client.create_collection(settings.collection, metadata={"hnsw:space": "cosine"})
    step = 1000
    for i in range(0, len(chunks), step):
        batch = chunks[i : i + step]
        col.add(
            ids=[c.id for c in batch],
            embeddings=vectors[i : i + step].tolist(),
            documents=[c.text for c in batch],
            metadatas=[
                {"source": c.source, "page_start": c.page_start, "page_end": c.page_end}
                for c in batch
            ],
        )
    print(f"Indexed {col.count()} chunks -> {settings.chroma_dir}")


if __name__ == "__main__":
    build_index()
