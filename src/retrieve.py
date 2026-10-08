"""Step 3 - Retrieve: vector search, BM25, hybrid fusion (RRF) and optional cross-encoder rerank."""
import json
import re
from dataclasses import dataclass
from functools import cached_property

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import settings

# Keep part numbers / specs like "6205-2RS", "M8x1.25", "3.5" as single tokens
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Hit:
    id: str
    text: str
    source: str
    page_start: int
    page_end: int
    score: float = 0.0

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


class Retriever:
    """mode: 'vector' | 'bm25' | 'hybrid'.  rerank: apply cross-encoder to the fused candidates."""

    def __init__(self, candidates: int = 20):
        self.candidates = candidates
        with open(settings.chunks_path, encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.col = client.get_collection(settings.collection)

    @cached_property
    def embedder(self) -> SentenceTransformer:
        return SentenceTransformer(settings.embed_model, device=settings.query_device)

    @cached_property
    def reranker(self) -> CrossEncoder:
        return CrossEncoder(settings.rerank_model, device=settings.query_device)

    def _hit(self, cid: str, score: float) -> Hit:
        c = self.by_id[cid]
        return Hit(cid, c["text"], c["source"], c["page_start"], c["page_end"], float(score))

    def vector_search(self, query: str, n: int) -> list[Hit]:
        q = self.embedder.encode(settings.query_prefix + query, normalize_embeddings=True)
        res = self.col.query(query_embeddings=[q.tolist()], n_results=n)
        return [self._hit(cid, 1 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]

    def bm25_search(self, query: str, n: int) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:n]
        return [self._hit(self.chunks[i]["id"], scores[i]) for i in top if scores[i] > 0]

    @staticmethod
    def rrf(*ranked_lists: list[Hit], k: int = 60) -> list[Hit]:
        """Reciprocal Rank Fusion: robust to the very different score scales of BM25 and cosine."""
        fused: dict[str, Hit] = {}
        for hits in ranked_lists:
            for rank, h in enumerate(hits):
                if h.id not in fused:
                    fused[h.id] = Hit(**{**h.__dict__, "score": 0.0})
                fused[h.id].score += 1.0 / (k + rank + 1)
        return sorted(fused.values(), key=lambda h: h.score, reverse=True)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self.reranker.predict([(query, h.text) for h in hits])
        for h, s in zip(hits, scores):
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def search(self, query: str, k: int = 5, mode: str = "hybrid", rerank: bool = True) -> list[Hit]:
        n = self.candidates
        if mode == "vector":
            hits = self.vector_search(query, n)
        elif mode == "bm25":
            hits = self.bm25_search(query, n)
        elif mode == "hybrid":
            hits = self.rrf(self.vector_search(query, n), self.bm25_search(query, n))[:n]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        if rerank:
            hits = self.rerank(query, hits)
        return hits[:k]
