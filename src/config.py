"""
Centralized configuration, loaded from environment variables / .env.

Keeping every tunable in one place is a small thing, but it's the difference
between "works on my machine" and something a teammate can pick up without
reading the whole codebase.
"""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Ollama (local LLM + embeddings, no API key needed) ---
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # --- Chroma ---
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "support_kb")

    # --- Langfuse (self-hosted) ---
    LANGFUSE_PUBLIC_KEY: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: str | None = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    # --- RAG tuning ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))

    @property
    def langfuse_enabled(self) -> bool:
        """Tracing is optional: the app still runs without Langfuse keys set,
        it just won't be observed. This makes local dev/testing easier."""
        return bool(self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)


config = Config()
