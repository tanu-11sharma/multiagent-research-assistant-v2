"""
FastAPI app exposing the multi-agent research pipeline over HTTP.

POST /research  { "question": "..." } -> a short, cited research brief
GET  /health     -> liveness + knowledge base stats
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.graph import run_research
from app.knowledge_base import KnowledgeBase

app = FastAPI(
    title="Multi-Agent Research Assistant (v2, critic loop)",
    description=(
        "A LangGraph planner -> searcher -> writer -> critic pipeline that "
        "produces a short, cited research brief from a bundled synthetic "
        "knowledge base. The critic agent grounds every claim against its "
        "cited source and sends weak sections back for revision before "
        "approving the final brief. Runs fully offline, no API keys."
    ),
    version="2.0.0",
)

_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


class ResearchRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, examples=["How does Python's GIL affect concurrency and how do distributed systems handle consensus?"]
    )


class SectionResponse(BaseModel):
    sub_question: str
    text: str
    doc_id: str | None
    chunk_index: int | None
    score: float | None
    revised: bool


class ResearchResponse(BaseModel):
    question: str
    approved: bool
    revision_count: int
    critique_notes: list[str]
    sections: list[SectionResponse]


@app.get("/health")
def health() -> dict:
    kb = get_kb()
    return {
        "status": "ok",
        "documents_loaded": kb.num_documents(),
        "chunks_indexed": kb.num_chunks(),
        "topics": kb.doc_ids(),
    }


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    kb = get_kb()
    result = run_research(request.question, kb)

    sections = [
        SectionResponse(
            sub_question=s.sub_question,
            text=s.text,
            doc_id=s.citation.doc_id if s.citation else None,
            chunk_index=s.citation.chunk_index if s.citation else None,
            score=s.citation.score if s.citation else None,
            revised=s.revised,
        )
        for s in result.get("sections", [])
    ]

    return ResearchResponse(
        question=request.question,
        approved=result.get("approved", False),
        revision_count=result.get("revision_count", 0),
        critique_notes=result.get("critique_notes", []),
        sections=sections,
    )
