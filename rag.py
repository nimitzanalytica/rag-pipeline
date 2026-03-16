"""
rag.py — Core RAG pipeline

Flow:
  1. ingest_document()  — load → chunk → embed → store in ChromaDB
  2. query()            — embed query → retrieve chunks → LLM generates answer

LLM provider is controlled by LLM_PROVIDER in .env:
  LLM_PROVIDER=ollama   — free, runs locally, no API key needed
  LLM_PROVIDER=claude   — Anthropic API (requires ANTHROPIC_API_KEY)
"""

import hashlib
import time
from pathlib import Path
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


# ---------------------------------------------------------------------------
# Singletons — initialised once, reused across requests
# ---------------------------------------------------------------------------

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Local sentence-transformer embeddings. No API key required."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def _get_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Persistent ChromaDB collection."""
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def _get_llm():
    """
    Return the configured LLM based on LLM_PROVIDER in .env.

    ollama → ChatOllama (local, free, private)
    claude → ChatAnthropic (Anthropic API, requires key)
    """
    if settings.llm_provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.max_tokens,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            num_predict=settings.max_tokens,
        )


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def _load_file(file_path: Path) -> list[Document]:
    """Load a PDF, Word, or plain-text file into LangChain Documents."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .docx, .txt, or .md")
    return loader.load()


def _chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _document_id(file_path: Path) -> str:
    """
    Stable document ID derived from filename + file content hash.
    Used to detect duplicates and to scope deletions.
    """
    content_hash = hashlib.md5(file_path.read_bytes()).hexdigest()[:8]
    return f"{file_path.stem}_{content_hash}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_document(file_path: Path) -> dict:
    """
    Ingest a document into the vector store.

    Returns metadata about the ingestion: doc_id, chunk count, etc.
    Raises ValueError if the document was already ingested.
    """
    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings)

    doc_id = _document_id(file_path)

    # Duplicate check
    existing = vectorstore.get(where={"doc_id": doc_id})
    if existing and existing.get("ids"):
        raise ValueError(
            f"Document '{file_path.name}' is already ingested (doc_id={doc_id}). "
            "Delete it first if you want to re-ingest."
        )

    # Load → chunk → embed → store
    raw_docs = _load_file(file_path)
    chunks = _chunk_documents(raw_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_id": doc_id,
            "filename": file_path.name,
            "chunk_index": i,
            "ingested_at": int(time.time()),
        })

    vectorstore.add_documents(chunks)

    return {
        "doc_id": doc_id,
        "filename": file_path.name,
        "chunks_created": len(chunks),
        "pages": len(raw_docs),
    }


def query_documents(question: str, doc_id: Optional[str] = None) -> dict:
    """
    Answer a question using retrieved context from the vector store.

    Args:
        question: Natural language question from the user.
        doc_id:   If provided, restrict retrieval to a single document.

    Returns:
        answer, source chunks used, retrieval metadata, and provider used.
    """
    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings)
    llm = _get_llm()

    # Optional per-document filtering
    search_kwargs: dict = {"k": settings.retrieval_k}
    if doc_id:
        search_kwargs["filter"] = {"doc_id": doc_id}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    relevant_chunks = retriever.invoke(question)

    if not relevant_chunks:
        return {
            "answer": "I couldn't find any relevant content in the ingested documents to answer that question.",
            "sources": [],
            "chunks_retrieved": 0,
            "provider": settings.llm_provider,
        }

    # Build context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        context_parts.append(
            f"[Excerpt {i} — {chunk.metadata.get('filename', 'unknown')}]\n{chunk.page_content}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on the "
        "provided document excerpts. Give a single, concise answer only. "
        "Do not reference excerpt numbers, list sources, or repeat the same answer multiple times. "
        "If the answer is not in the excerpts, say so clearly."
    )
    user_prompt = f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    sources = [
        {
            "filename": c.metadata.get("filename"),
            "doc_id": c.metadata.get("doc_id"),
            "chunk_index": c.metadata.get("chunk_index"),
            "page": c.metadata.get("page"),
        }
        for c in relevant_chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
        "chunks_retrieved": len(relevant_chunks),
        "provider": settings.llm_provider,
    }


def list_documents() -> list[dict]:
    """Return a deduplicated list of all ingested documents."""
    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings)

    all_metadata = vectorstore.get()["metadatas"]
    if not all_metadata:
        return []

    seen: dict[str, dict] = {}
    for meta in all_metadata:
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("filename"),
                "ingested_at": meta.get("ingested_at"),
            }

    return list(seen.values())


def delete_document(doc_id: str) -> dict:
    """
    Remove all chunks belonging to a document from the vector store.
    Returns the number of chunks deleted.
    """
    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings)

    existing = vectorstore.get(where={"doc_id": doc_id})
    chunk_ids = existing.get("ids", [])

    if not chunk_ids:
        raise ValueError(f"No document found with doc_id='{doc_id}'")

    vectorstore.delete(ids=chunk_ids)

    return {"doc_id": doc_id, "chunks_deleted": len(chunk_ids)}
