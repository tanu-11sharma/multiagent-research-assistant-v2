"""
The four agents in the pipeline: planner, searcher, writer, critic.

Each agent is a plain function operating on a shared state dict, matching
the node signature LangGraph expects (state -> partial state update). None
of them call an external LLM: the "planning" and "writing" steps are rule
based / extractive, which keeps the whole pipeline deterministic, testable,
and runnable with zero API keys. Swapping any node for a real LLM call is a
drop-in change (see README).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.knowledge_base import Citation, KnowledgeBase

# Keyword sets used by the planner to recognize which knowledge-base topics
# a question touches, so it can decompose a multi-topic question into one
# sub-question per topic instead of searching everything at once.
TOPIC_KEYWORDS: dict[str, set[str]] = {
    "distributed_systems": {
        "distributed", "cap theorem", "consistency", "replication",
        "consensus", "raft", "paxos", "partition", "leader election",
    },
    "python_basics": {
        "python", "gil", "interpreter", "dynamic typing", "pip", "poetry",
    },
    "vector_databases": {
        "embedding", "embeddings", "vector database", "vector databases",
        "rag", "retrieval-augmented", "ann", "hnsw", "qdrant", "pinecone",
    },
    "rest_apis": {
        "rest", "api", "http", "idempotent", "idempotency", "status code",
        "versioning", "endpoint",
    },
    "testing_practices": {
        "test", "testing", "unit test", "integration test", "tdd",
        "end-to-end", "test pyramid",
    },
}


@dataclass
class Section:
    sub_question: str
    text: str
    citation: Citation | None
    revised: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = asdict(self.citation) if self.citation else None
        return d


@dataclass
class Critique:
    approved: bool
    issues: list[int]  # indices into sections that need revision
    notes: list[str]


def plan(question: str, min_score: float = 0.0) -> list[str]:
    """Decompose a question into 1-3 sub-questions.

    If the question's wording matches more than one known topic, one
    sub-question is generated per matched topic (grounded decomposition).
    Otherwise the original question is used as the single sub-question.
    """
    q_lower = question.lower()
    matched_topics = [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(kw in q_lower for kw in keywords)
    ]

    if len(matched_topics) <= 1:
        return [question.strip()]

    topic_labels = {
        "distributed_systems": "distributed systems",
        "python_basics": "Python",
        "vector_databases": "vector databases / RAG",
        "rest_apis": "REST API design",
        "testing_practices": "software testing",
    }
    return [
        f"{question.strip()} (focus: {topic_labels[topic]})"
        for topic in matched_topics
    ]


def search(
    sub_questions: list[str],
    kb: KnowledgeBase,
    top_k: int = 2,
) -> dict[str, list[Citation]]:
    return {sq: kb.retrieve(sq, top_k=top_k) for sq in sub_questions}


def _best_sentence(text: str, query: str) -> str:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if len(sentences) <= 1:
        return sentences[0] if sentences else text
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform(sentences)
        q_vec = vec.transform([query])
        scores = cosine_similarity(q_vec, matrix)[0]
        return sentences[int(scores.argmax())]
    except ValueError:
        return sentences[0]


def write(
    sub_questions: list[str],
    research_notes: dict[str, list[Citation]],
) -> list[Section]:
    sections = []
    for sq in sub_questions:
        citations = research_notes.get(sq, [])
        if not citations:
            sections.append(
                Section(sub_question=sq, text="No relevant source found.", citation=None)
            )
            continue
        best = citations[0]
        sentence = _best_sentence(best.snippet, sq)
        text = f"{sentence} (source: {best.doc_id}, chunk #{best.chunk_index})"
        sections.append(Section(sub_question=sq, text=text, citation=best))
    return sections


def critique(sections: list[Section], min_score: float = 0.12) -> Critique:
    """Grounding + confidence check on each section.

    A section fails if it has no citation, its citation's similarity score
    is below min_score (weak evidence for the claim), or the sentence used
    in the section text is not actually contained in the cited snippet
    (a hallucination/grounding check).
    """
    issues: list[int] = []
    notes: list[str] = []
    for i, section in enumerate(sections):
        if section.citation is None:
            issues.append(i)
            notes.append(f"Section {i} ('{section.sub_question}') has no supporting citation.")
            continue
        if section.citation.score < min_score:
            issues.append(i)
            notes.append(
                f"Section {i} citation score {section.citation.score} is below "
                f"confidence threshold {min_score}."
            )
            continue
        # Grounding check: the sentence quoted in the answer text must be a
        # substring of the cited snippet it claims to come from.
        quoted_sentence = section.text.split(" (source:")[0].strip()
        if quoted_sentence not in section.citation.snippet:
            issues.append(i)
            notes.append(f"Section {i} text is not grounded in its cited snippet.")

    return Critique(approved=len(issues) == 0, issues=issues, notes=notes)


def revise(
    sections: list[Section],
    sub_questions: list[str],
    kb: KnowledgeBase,
    issues: list[int],
    top_k: int = 2,
) -> list[Section]:
    """Ask the searcher for the next-best citation for each flagged section
    (excluding the one already tried) and rewrite that section.
    """
    revised = list(sections)
    for i in issues:
        sq = sub_questions[i]
        already_tried = frozenset(
            {revised[i].citation.chunk_id} if revised[i].citation else set()
        )
        alternatives = kb.retrieve(sq, top_k=top_k, exclude_chunk_ids=already_tried)
        if not alternatives:
            continue
        best = alternatives[0]
        sentence = _best_sentence(best.snippet, sq)
        text = f"{sentence} (source: {best.doc_id}, chunk #{best.chunk_index})"
        revised[i] = Section(sub_question=sq, text=text, citation=best, revised=True)
    return revised
