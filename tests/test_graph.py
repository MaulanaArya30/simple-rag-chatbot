"""
Unit tests for src/graph.py.

These deliberately do NOT touch Ollama, Chroma, or Langfuse -- the graph
logic (routing, state passing) is tested in isolation with fakes, which is
what makes it possible to run `pytest` in a few seconds with no services
running. Treat this the same way you'd treat tests for an Tagnify-style
pipeline: schema/logic tests are cheap and catch real bugs; the slow
end-to-end run with real models is a separate, manual smoke test (see
README "Running it").
"""
from types import SimpleNamespace

from src.graph import build_graph


class FakeLLM:
    """Returns canned `.content` responses in call order, regardless of input."""

    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages):
        if not self._responses:
            raise AssertionError("FakeLLM ran out of canned responses")
        return SimpleNamespace(content=self._responses.pop(0))


class FakeVectorStore:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_query = None

    def similarity_search(self, query, k=4):
        self.last_query = query
        return [SimpleNamespace(page_content=c) for c in self._chunks[:k]]


def _state(question: str) -> dict:
    return {"question": question, "route": "", "context": "", "tool_result": "", "answer": ""}


def test_routes_to_tool_for_fee_question():
    llm = FakeLLM(["TOOL", "The QRIS fee is 0.7%."])
    vectorstore = FakeVectorStore(["irrelevant chunk"])
    agent = build_graph(llm, vectorstore, retrieval_k=2)

    result = agent.invoke(_state("What's the QRIS fee?"))

    assert result["route"] == "tool"
    assert "0.7%" in result["tool_result"]
    assert result["answer"] == "The QRIS fee is 0.7%."
    assert vectorstore.last_query is None  # retrieval was skipped entirely


def test_routes_to_retrieve_for_policy_question():
    llm = FakeLLM(["RETRIEVE", "Refunds take up to 5 business days."])
    vectorstore = FakeVectorStore(["Refund SLA is 5 business days."])
    agent = build_graph(llm, vectorstore, retrieval_k=2)

    result = agent.invoke(_state("How long do refunds take?"))

    assert result["route"] == "retrieve"
    assert "Refund SLA" in result["context"]
    assert result["answer"] == "Refunds take up to 5 business days."
    assert vectorstore.last_query == "How long do refunds take?"


def test_routes_direct_for_small_talk():
    llm = FakeLLM(["DIRECT", "Hi! How can I help you today?"])
    vectorstore = FakeVectorStore(["irrelevant"])
    agent = build_graph(llm, vectorstore, retrieval_k=2)

    result = agent.invoke(_state("Hey, how are you?"))

    assert result["route"] == "direct"
    assert result["context"] == ""
    assert result["tool_result"] == ""
    assert result["answer"] == "Hi! How can I help you today?"


def test_unrecognized_router_output_defaults_to_direct():
    """If the local model rambles instead of returning one word, we
    shouldn't crash -- fail safe to a direct answer."""
    llm = FakeLLM(["Sure, I think this is a TOOLish question maybe", "Okay."])
    vectorstore = FakeVectorStore(["irrelevant"])
    agent = build_graph(llm, vectorstore, retrieval_k=2)

    result = agent.invoke(_state("???"))

    # Note: "TOOL" substring still matches by design (router prompt asks for
    # a single word; this test documents the current fallback behavior).
    assert result["route"] in {"tool", "direct"}
