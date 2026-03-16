"""
main.py — FastAPI application

Run with:
    uvicorn main:app --reload

Interactive API docs available at:
    http://127.0.0.1:8000/docs   (Swagger UI)
    http://127.0.0.1:8000/redoc  (ReDoc)
"""

import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database
import rag
from config import settings


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Q&A API",
    description=(
        "A RAG (Retrieval-Augmented Generation) API that lets you upload documents "
        "and ask natural language questions about them. "
        "Powered by FastAPI, LangChain, ChromaDB, and Claude."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    """Initialize PostgreSQL table on startup."""
    database.init_db()


# ---------------------------------------------------------------------------
# Pydantic schemas — request / response shapes
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are the main topics covered in this document?",
                    "doc_id": None,
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks_retrieved: int


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_created: int
    pages: int
    message: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: Optional[str]
    ingested_at: Optional[int]


class DeleteResponse(BaseModel):
    doc_id: str
    chunks_deleted: int
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "Document Q&A API is running."}


@app.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload and ingest a document",
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, DOCX, TXT, or MD file. The document will be:
    - Chunked into overlapping segments
    - Embedded using a local sentence-transformer model
    - Stored in ChromaDB for retrieval

    Returns a doc_id you can use to scope queries to this document.
    """
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(allowed_extensions)}",
        )

    save_path = settings.upload_dir / file.filename
    try:
        with save_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    try:
        result = rag.ingest_document(save_path)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )

    return IngestResponse(
        **result,
        message=f"Successfully ingested '{file.filename}' into {result['chunks_created']} chunks.",
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Q&A"],
    summary="Ask a question about ingested documents",
)
def query(request: QueryRequest):
    """
    Submit a natural language question. The LLM will answer using only content
    retrieved from your ingested documents. Every query is logged to PostgreSQL.

    Optionally pass a doc_id to restrict the search to a single document.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    try:
        result = rag.query_documents(
            question=request.question,
            doc_id=request.doc_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )

    # Log query to PostgreSQL — failure never breaks the response
    database.log_query(
        question=request.question,
        answer=result["answer"],
        provider=result.get("provider", "unknown"),
        chunks_retrieved=result["chunks_retrieved"],
        sources=result["sources"],
        doc_id=request.doc_id,
    )

    return QueryResponse(**result)


@app.get(
    "/documents",
    response_model=list[DocumentInfo],
    tags=["Documents"],
    summary="List all ingested documents",
)
def list_documents():
    """
    Returns a list of all documents currently stored in the vector database,
    with their doc_id, filename, and ingestion timestamp.
    """
    try:
        docs = rag.list_documents()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve document list: {str(e)}",
        )
    return docs


@app.delete(
    "/documents/{doc_id}",
    response_model=DeleteResponse,
    tags=["Documents"],
    summary="Delete a document from the vector store",
)
def delete_document(doc_id: str):
    """
    Remove all chunks for the given doc_id from ChromaDB.
    The document will no longer be searchable after deletion.
    """
    try:
        result = rag.delete_document(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {str(e)}",
        )

    return DeleteResponse(
        **result,
        message=f"Deleted {result['chunks_deleted']} chunks for doc_id='{doc_id}'.",
    )
