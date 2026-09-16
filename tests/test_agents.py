from app.agents import critique, plan, revise, search, write
from app.knowledge_base import KnowledgeBase


def test_plan_single_topic_returns_one_subquestion():
    sub_qs = plan("What is the CAP theorem?")
    assert len(sub_qs) == 1


def test_plan_multi_topic_question_decomposes():
    sub_qs = plan(
        "How does the Python GIL affect concurrency and how does distributed "
        "systems consensus like Raft work?"
    )
    assert len(sub_qs) == 2
    joined = " ".join(sub_qs)
    assert "Python" in joined
    assert "distributed systems" in joined


def test_search_returns_citations_per_subquestion():
    kb = KnowledgeBase()
    sub_qs = plan("What is a vector database?")
    notes = search(sub_qs, kb)
    assert sub_qs[0] in notes
    assert len(notes[sub_qs[0]]) > 0


def test_write_produces_grounded_sections():
    kb = KnowledgeBase()
    sub_qs = ["What is the CAP theorem?"]
    notes = search(sub_qs, kb)
    sections = write(sub_qs, notes)
    assert len(sections) == 1
    assert sections[0].citation is not None
    assert "(source:" in sections[0].text


def test_critique_approves_well_grounded_sections():
    kb = KnowledgeBase()
    sub_qs = ["What is the CAP theorem in distributed systems?"]
    notes = search(sub_qs, kb)
    sections = write(sub_qs, notes)
    result = critique(sections)
    assert result.approved
    assert result.issues == []


def test_critique_flags_ungrounded_section():
    from app.agents import Section
    from app.knowledge_base import Citation

    bad_citation = Citation(doc_id="python_basics", chunk_index=0, snippet="Python is great.", score=0.5)
    bad_section = Section(
        sub_question="irrelevant",
        text="This sentence never appears in the snippet. (source: python_basics, chunk #0)",
        citation=bad_citation,
    )
    result = critique([bad_section])
    assert not result.approved
    assert 0 in result.issues


def test_revise_picks_an_alternative_citation():
    from app.agents import Section
    from app.knowledge_base import Citation

    kb = KnowledgeBase()
    sub_qs = ["distributed systems consistency"]
    original_citations = kb.retrieve(sub_qs[0], top_k=1)
    section = Section(sub_question=sub_qs[0], text="placeholder", citation=original_citations[0])

    revised = revise([section], sub_qs, kb, issues=[0])
    assert revised[0].revised is True
    assert revised[0].citation.chunk_id != original_citations[0].chunk_id
