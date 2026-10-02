"""
The agent graph.

This is intentionally small but non-trivial: instead of naive RAG (always
retrieve, always generate), a router node first decides HOW to answer:

    route_query
       |-- "tool"     -> call_tool      -> generate
       |-- "retrieve" -> retrieve(Chroma) -> generate
       `-- "direct"   -> generate directly

That routing decision is itself an LLM call, which is the part that makes
this an *agent workflow* rather than a fixed pipeline -- and it's also why
tracing matters: each branch produces a different number of LLM calls and a
different cost, which is exactly what Langfuse/LangSmith make visible.

`llm` and `vectorstore` are injected rather than constructed here, so the
graph can be unit-tested with fakes (see tests/test_graph.py) without needing
a running Ollama server or a populated Chroma collection.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.tools import lookup_transaction_fee


class GraphState(TypedDict):
    question: str
    route: str
    context: str
    tool_result: str
    answer: str
    token_usage: str


ROUTER_SYSTEM_PROMPT = (
    "You are a routing classifier for a fintech support agent. "
    "Given a user question, respond with exactly one word:\n"
    "- TOOL if the question asks for a specific transaction fee amount "
    "(e.g. QRIS fee, cash-out fee, checkout fee)\n"
    "- RETRIEVE if the question needs policy or FAQ knowledge "
    "(KYC, refunds, settlement, top-up, general fee policy)\n"
    "- DIRECT if the question is small talk or needs no company knowledge\n"
    "Respond with only the single word, nothing else."
)

GENERATE_SYSTEM_PROMPT = (
    "You are a support assistant for a digital payments app. Answer the "
    "user's question using the provided context. The context sometimes "
    "uses internal terminology -- for example, 'MDR' (Merchant Discount "
    "Rate) IS the merchant fee. Connect such terms to the question instead "
    "of treating a wording difference as missing information. If the "
    "context genuinely does not contain the answer, say so rather than "
    "guessing."
)

def _accumulate_usage(state: GraphState, result) -> dict:
    """Pulls usage_metadata off an LLM response and adds it to whatever's
    already in state, so token_usage accumulates across both LLM calls
    (router + generate) instead of the second call overwriting the first."""
    usage = getattr(result, "usage_metadata", None) or {}
    existing = state.get("token_usage") or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": existing.get("input_tokens", 0) + usage.get("input_tokens", 0),
        "output_tokens": existing.get("output_tokens", 0) + usage.get("output_tokens", 0),
        "total_tokens": existing.get("total_tokens", 0) + usage.get("total_tokens", 0),
    }

def _make_router_node(llm):
    def route_query(state: GraphState) -> GraphState:
        messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=state["question"])]
        result = llm.invoke(messages)
        label = result.content.strip().upper()
        route = "tool" if "TOOL" in label else "retrieve" if "RETRIEVE" in label else "direct"
        return {**state, "route": route, "token_usage": _accumulate_usage(state, result)}  # ← added
    return route_query

def _make_tool_node():
    def call_tool(state: GraphState) -> GraphState:
        question = state["question"].lower()
        if "qris" in question:
            key = "qris"
        elif "checkout" in question:
            key = "checkout"
        elif "cash" in question or "withdraw" in question:
            key = "cashout_medium"
        else:
            key = "qris"
        return {**state, "tool_result": lookup_transaction_fee(key)}

    return call_tool

def _make_retrieve_node(vectorstore, k: int):
    def retrieve(state: GraphState) -> GraphState:
        docs = vectorstore.similarity_search(state["question"], k=k)
        context = "\n\n".join(d.page_content for d in docs)
        return {**state, "context": context}

    return retrieve

def _make_generate_node(llm):
    def generate(state: GraphState) -> GraphState:
        context_parts = [p for p in (state.get("context"), state.get("tool_result")) if p]
        context = "\n\n".join(context_parts) or "(no additional context)"
        messages = [SystemMessage(content=GENERATE_SYSTEM_PROMPT),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}")]
        result = llm.invoke(messages)
        return {**state, "answer": result.content, "token_usage": _accumulate_usage(state, result)}  # ← added
    return generate


def _routing_function(state: GraphState) -> Literal["tool", "retrieve", "direct"]:
    return state["route"]  # type: ignore[return-value]


def build_graph(llm, vectorstore, retrieval_k: int = 4):
    """Wire up and compile the agent graph."""
    graph = StateGraph(GraphState)

    graph.add_node("route_query", _make_router_node(llm))
    graph.add_node("call_tool", _make_tool_node())
    graph.add_node("retrieve", _make_retrieve_node(vectorstore, retrieval_k))
    graph.add_node("generate", _make_generate_node(llm))

    graph.set_entry_point("route_query")
    graph.add_conditional_edges(
        "route_query",
        _routing_function,
        {"tool": "call_tool", "retrieve": "retrieve", "direct": "generate"},
    )
    graph.add_edge("call_tool", "generate")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
