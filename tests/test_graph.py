from app.graph import run_research
from app.knowledge_base import KnowledgeBase


def test_run_research_single_topic_is_approved_without_revision():
    kb = KnowledgeBase()
    result = run_research("What is the CAP theorem?", kb)
    assert result["approved"] is True
    assert result["revision_count"] == 0
    assert len(result["sections"]) == 1
    assert result["sections"][0].citation is not None


def test_run_research_multi_topic_question_produces_multiple_sections():
    kb = KnowledgeBase()
    result = run_research(
        "How does the Python GIL affect concurrency and how does distributed "
        "systems consensus like Raft work?",
        kb,
    )
    assert len(result["sections"]) == 2
    doc_ids = {s.citation.doc_id for s in result["sections"] if s.citation}
    assert "python_basics" in doc_ids
    assert "distributed_systems" in doc_ids


def test_run_research_terminates_even_when_never_approved():
    from app import graph

    original_critique = graph.critique

    def always_fail(sections, min_score=0.12):
        result = original_critique(sections, min_score=999.0)
        return result

    graph.critique = always_fail
    try:
        kb = KnowledgeBase()
        result = run_research("What is the CAP theorem?", kb)
        # Must terminate within MAX_REVISIONS and report not approved.
        assert result["approved"] is False
        assert result["revision_count"] <= 2
    finally:
        graph.critique = original_critique
