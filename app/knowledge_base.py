"""
TF-IDF backed knowledge base shared by the searcher agent.

Loads a small set of plain-text documents, chunks them by paragraph, and
answers similarity queries via TF-IDF + cosine similarity. No external
embeddings API or LLM call is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#{self.chunk_index}"


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_index: int
    snippet: str
    score: float

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#{self.chunk_index}"


def _split_into_chunks(doc_id: str, raw_text: str) -> list[Chunk]:
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    return [Chunk(doc_id=doc_id, chunk_index=i, text=p) for i, p in enumerate(paragraphs)]


class KnowledgeBase:
    """Loads documents and answers TF-IDF similarity queries."""

    def __init__(self, docs_dir: Path = DOCS_DIR):
        self.docs_dir = docs_dir
        self.chunks: list[Chunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._load()

    def _load(self) -> None:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")
        for path in sorted(self.docs_dir.glob("*.txt")):
            self.chunks.extend(_split_into_chunks(path.stem, path.read_text(encoding="utf-8")))
        if not self.chunks:
            raise ValueError(f"No documents found in {self.docs_dir}")
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def num_documents(self) -> int:
        return len({c.doc_id for c in self.chunks})

    def num_chunks(self) -> int:
        return len(self.chunks)

    def doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self.chunks})

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        exclude_chunk_ids: frozenset[str] = frozenset(),
    ) -> list[Citation]:
        """Return the top_k most relevant chunks for query, as Citations.

        exclude_chunk_ids lets a caller ask for the *next best* alternatives,
        used by the critic/writer revision loop.
        """
        if self.vectorizer is None or self._matrix is None:
            raise RuntimeError("Knowledge base was not initialized correctly")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = scores.argsort()[::-1]

        citations = []
        for idx in ranked:
            chunk = self.chunks[idx]
            if chunk.chunk_id in exclude_chunk_ids:
                continue
            score = float(scores[idx])
            if score <= 0:
                continue
            citations.append(
                Citation(
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    snippet=chunk.text,
                    score=round(score, 4),
                )
            )
            if len(citations) >= top_k:
                break
        return citations
