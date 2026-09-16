# Multi-Agent Research Assistant v2 (critic / reflection loop)

A four-agent research pipeline -- **planner -> searcher -> writer -> critic**
-- built with [LangGraph](https://github.com/langchain-ai/langgraph), that
turns a question into a short, cited research brief drawn from a bundled
synthetic knowledge base. The critic agent grounds every claim against its
citation and can send weak sections back to the writer for revision before
approving the final brief.

This is a v2 twist on an earlier `multiagent-research-assistant` build in
this account: the original was a single-pass planner/searcher/writer
pipeline with no self-checking step. v2 adds a **critic node with a bounded
reflection loop** -- a common pattern in real agentic systems, where an
agent's own output is checked and revised before it's returned, rather than
trusting the first draft.

## Why this is relevant

Agentic AI and multi-agent orchestration (planner/worker/critic patterns,
reflection loops, LangGraph-style graphs) are one of the most common
architectures in production LLM applications right now. This project
implements that shape end-to-end -- including the self-correction loop --
without hiding the mechanics behind a hosted LLM API: every node is a plain,
testable Python function, and the "grounding check" the critic performs
(does the answer text actually appear in its cited source?) is a real,
inspectable rule rather than another opaque model call. Swapping any node
for a real LLM call (e.g. having the planner/writer call Claude or GPT
instead of using heuristics) is a drop-in change.

## How it works

1. **Planner** -- looks at the question and, if it touches more than one
   known topic in the knowledge base, splits it into one sub-question per
   topic. A single-topic question stays as one sub-question.
2. **Searcher** -- retrieves the most relevant passages for each
   sub-question from a TF-IDF index over the bundled documents (same
   retrieval approach as a RAG system, no external embeddings API needed).
3. **Writer** -- drafts one section per sub-question: the best-matching
   sentence from the top-cited passage, tagged with its source document and
   chunk number.
4. **Critic** -- checks every section for three things: it has a citation,
   the citation's similarity score clears a confidence threshold, and the
   sentence quoted in the section actually appears in its cited snippet
   (a hallucination/grounding check). Sections that fail get sent back to
   be rewritten with the next-best available citation, up to 2 revision
   rounds, before the brief is returned with an `approved` flag either way.

```
planner -> searcher -> writer -> critic --(approved / out of revisions)--> done
                                     ^                |
                                     +--(needs fix)----+
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8001
```

API at `http://127.0.0.1:8001`, interactive docs at
`http://127.0.0.1:8001/docs`.

## Test

```bash
pytest -v
```

14 tests cover the planner's topic decomposition, the critic's grounding
checks (including a deliberately ungrounded section to make sure it gets
flagged), the revision loop picking a genuinely different citation, the
compiled LangGraph pipeline end to end, and the HTTP API.

## Example usage

```bash
curl -X POST http://127.0.0.1:8001/research \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the Python GIL affect concurrency and how does distributed systems consensus like Raft work?"}'
```

This question touches two known topics, so the planner splits it into two
sub-questions, and the response contains two cited sections -- one grounded
in `python_basics`, one in `distributed_systems`.

## Project structure

```
app/
  knowledge_base.py   TF-IDF chunking + retrieval (the "searcher" backend)
  agents.py            plan / search / write / critique / revise functions
  graph.py             LangGraph StateGraph wiring the four agents together
  main.py              FastAPI app exposing POST /research, GET /health
data/docs/             Sample knowledge base (5 short synthetic reference docs)
tests/                 pytest suite: agents, compiled graph, HTTP API
```

## Notes

- All documents in `data/docs/` are synthetic reference notes on generic
  software engineering topics -- not proprietary or real-world data.
- Demo/reference implementation: in-memory index, no persistence or auth,
  and the "LLM" steps are deterministic heuristics rather than a real model
  call, by design, so the whole thing runs offline with zero API keys.
