"""
CLI entry point.

Usage:
    python app.py "What's the QRIS merchant fee?"
    python app.py                       # interactive loop

Before running this, you need:
    1. Ollama running locally with the chat + embedding models pulled
       (see README "Setup").
    2. The vector store ingested at least once: python -m src.ingest
    3. (Optional but the point of this project) Langfuse running locally
       and its keys set in .env -- see README "Setting up Langfuse".
"""
import sys

from langchain_ollama import ChatOllama

from src.config import config
from src.graph import build_graph
from src.ingest import load_existing_vectorstore


def _build_callbacks():
    if not config.langfuse_enabled:
        print(
            "[info] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set - "
            "running without tracing. See .env.example to enable it.\n"
        )
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def _empty_state(question: str) -> dict:
    return {
        "question": question, "route": "", "context": "", "tool_result": "", "answer": "",
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},  # ← added
    }


def main():
    llm = ChatOllama(
        model=config.OLLAMA_CHAT_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
    )
    vectorstore = load_existing_vectorstore()
    agent = build_graph(llm, vectorstore, retrieval_k=config.RETRIEVAL_K)
    callbacks = _build_callbacks()

    def ask(question: str) -> str:
        result = agent.invoke(_empty_state(question), config={"callbacks": callbacks})
        print(f"[debug] route={result['route']!r}")
        print(f"[debug] context={result['context']!r}")
        print(f"[debug] tool_result={result['tool_result']!r}")
        print(f"[debug] token_usage={result['token_usage']!r}")   # ← added

        return result["answer"]

    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:])))
        return

    print("Monee RAG agent demo. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print(f"Agent: {ask(question)}\n")


if __name__ == "__main__":
    main()
