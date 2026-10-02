"""
One-time (or re-runnable) ingestion script.

Reads every .txt file in data/, splits it into overlapping chunks, embeds each
chunk with a local Ollama embedding model, and writes the vectors into a
persistent ChromaDB collection on disk.

Run with:  python -m src.ingest
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_and_chunk():
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()
    if not raw_docs:
        raise FileNotFoundError(
            f"No .txt files found in {DATA_DIR}. Add some documents first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Loaded {len(raw_docs)} documents -> split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks) -> Chroma:
    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    print(f"Persisted vector store to {config.CHROMA_PERSIST_DIR}")
    return vectorstore


def load_existing_vectorstore() -> Chroma:
    """Used by the app at query time, instead of re-ingesting every run."""
    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


if __name__ == "__main__":
    chunks = load_and_chunk()
    build_vectorstore(chunks)
