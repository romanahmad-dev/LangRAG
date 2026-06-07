"""
LangRAG — Core RAG Pipeline
Load → Chunk → Embed → Store → Retrieve
Built with LangChain + ChromaDB + HuggingFace sentence-transformers.
"""

import os
from typing import List, Optional

from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

from config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)


def load_document(file_path: str) -> List[Document]:
    """Load a PDF or TXT file using the appropriate LangChain loader."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return PyMuPDFLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks via RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a local HuggingFace sentence-transformer embedding model (free)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(
    chunks: List[Document],
    embedder: HuggingFaceEmbeddings,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """Embed chunks and persist them in ChromaDB."""
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_dir,
    )
    db.persist()
    return db


def retrieve_context(db: Chroma, query: str, k: int = TOP_K) -> tuple[str, List[Document]]:
    """
    Retrieve top-k most relevant chunks for a query.
    Returns both the joined context string and the raw doc list.
    """
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])
    return context, docs


def ingest(file_path: str, persist_dir: str = CHROMA_PERSIST_DIR) -> tuple[Chroma, dict]:
    """
    Full ingestion pipeline: Load → Chunk → Embed → Store.
    Returns (vectordb, stats_dict).
    """
    docs   = load_document(file_path)
    chunks = chunk_documents(docs)
    embedder = get_embedding_model()
    db = build_vector_store(chunks, embedder, persist_dir)

    stats = {
        "pages":  len(docs),
        "chunks": len(chunks),
        "model":  EMBEDDING_MODEL,
        "store":  persist_dir,
    }
    return db, stats
