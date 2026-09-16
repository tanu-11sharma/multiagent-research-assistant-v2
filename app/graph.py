"""
LangGraph wiring for the planner -> searcher -> writer -> critic pipeline.

The critic can route back to the writer for a bounded number of revisions
(a reflection loop) before the graph ends -- a common agentic pattern for
improving answer quality without a human in the loop.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents import Section, critique, plan, revise, search, write
from app.knowledge_base import KnowledgeBase

MAX_REVISIONS = 2


class ResearchState(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    research_notes: dict
    sections: list[Section]
    approved: bool
    critique_notes: list[str]
    revision_count: int


def _planner_node(state: ResearchState) -> dict:
    return {"sub_questions": plan(state["question"])}


def _make_searcher_node(kb: KnowledgeBase):
    def _searcher_node(state: ResearchState) -> dict:
        return {"research_notes": search(state["sub_questions"], kb)}

    return _searcher_node


def _writer_node(state: ResearchState) -> dict:
    sections = write(state["sub_questions"], state["research_notes"])
    return {"sections": sections}


def _make_critic_node(kb: KnowledgeBase):
    def _critic_node(state: ResearchState) -> dict:
        result = critique(state["sections"])
        revision_count = state.get("revision_count", 0)

        if result.approved or revision_count >= MAX_REVISIONS:
            return {
                "approved": result.approved,
                "critique_notes": result.notes,
            }

        revised_sections = revise(
            state["sections"], state["sub_questions"], kb, result.issues
        )
        return {
            "sections": revised_sections,
            "approved": False,
            "critique_notes": result.notes,
            "revision_count": revision_count + 1,
        }

    return _critic_node


def _route_after_critic(state: ResearchState) -> str:
    if state.get("approved"):
        return END
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return END
    return "writer_ack"


def build_graph(kb: KnowledgeBase | None = None):
    """Builds and compiles the multi-agent research graph.

    Graph shape:

        planner -> searcher -> writer -> critic --(approved or out of
                                            ^          revisions)--> END
                                            |
                                            +---(needs revision)-----+
                                    (critic already rewrote the
                                     flagged sections in-place;
                                     writer_ack just loops back to
                                     critic to re-check them)
    """
    kb = kb or KnowledgeBase()
    graph = StateGraph(ResearchState)

    graph.add_node("planner", _planner_node)
    graph.add_node("searcher", _make_searcher_node(kb))
    graph.add_node("writer", _writer_node)
    graph.add_node("critic", _make_critic_node(kb))
    # No-op pass-through node: the critic already revised the flagged
    # sections, this node just gives the graph a place to route back to
    # before re-entering the critic for another approval check.
    graph.add_node("writer_ack", lambda state: {})

    graph.set_entry_point("planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", _route_after_critic, {"writer_ack": "writer_ack", END: END})
    graph.add_edge("writer_ack", "critic")

    return graph.compile()


def run_research(question: str, kb: KnowledgeBase | None = None) -> ResearchState:
    app_graph = build_graph(kb)
    initial_state: ResearchState = {"question": question, "revision_count": 0}
    return app_graph.invoke(initial_state)
