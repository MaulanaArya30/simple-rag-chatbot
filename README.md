# Simple RAG Chatbot

A small LangGraph + LangChain chatbot that answers customer-support questions for a fictional e-wallet. It routes each question to one of three paths: a tool call, retrieval from a ChromaDB vector store, or a direct answer. Every LLM call can be traced in Langfuse to see cost and token usage.

It runs fully locally on Ollama, so no paid API key is needed.

```
                    ┌────────────────┐
   User question ──►│  route_query   │  ← 1 LLM call: classify the question
                    └───────┬────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
        route="tool"   route="retrieve"  route="direct"
            │               │               │
     ┌─────────────┐  ┌─────────────┐       │
     │  call_tool  │  │  retrieve   │       │
     └──────┬──────┘  └──────┬──────┘       │
            └────────┬───────┴──────────────┘
                      ▼
               ┌─────────────┐
               │  generate   │  ← 1 LLM call: produce the final answer
               └──────┬──────┘
                      ▼
                   Answer
```

## Stack

- **LangGraph** for the routing graph (`src/graph.py`)
- **LangChain + Ollama** for the chat model (`llama3.2:3b`) and embeddings (`nomic-embed-text`)
- **ChromaDB** as the persistent vector store (`src/ingest.py`)
- **Langfuse** (optional, self-hosted) for tracing and token/cost tracking
- A mock fee-lookup tool (`src/tools.py`) to show the tool-calling path

The knowledge base in `data/` is AI-generated demo content (fees, top-ups, KYC, refunds, merchant settlement). It is not real company policy.

## Setup

```bash
# 1. Install dependencies
poetry install

# 2. Pull the local models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 3. Configure (Langfuse keys are optional)
cp .env.example .env

# 4. Build the vector store
poetry run python -m src.ingest
```

## Usage

```bash
poetry run python app.py "What's the QRIS merchant fee?"   # single question
poetry run python app.py                                   # interactive loop
```

## Tests

```bash
poetry run pytest
```

## Setting up Langfuse

Run Langfuse locally (see the [self-hosting docs](https://langfuse.com/self-hosting)), create a project, and put its public and secret keys in `.env`. Without the keys, the app still runs but doesn't trace.
