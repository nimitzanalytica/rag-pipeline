# Document Q&A API

A **RAG (Retrieval-Augmented Generation)** API that lets you upload documents and ask natural language questions about them. Built as a portfolio project demonstrating a production-style AI pipeline.

## Use Cases

This API is designed for any domain where people need to query large volumes of documents with natural language. Some examples:

| Domain | Example |
|---|---|
| 🎓 **EdTech** | Upload a course syllabus, textbook chapter, or lecture notes — students ask questions and get grounded, cited answers instantly |
| 🎓 **EdTech** | Ingest an entire curriculum — instructors query it to find alignment between learning objectives and assessments |
| 💊 **Healthcare** | Upload clinical guidelines — practitioners query drug protocols without reading 40-page PDFs |
| ⚖️ **Legal** | Ingest case files or contracts — lawyers ask plain-English questions about specific clauses |
| 🏢 **Enterprise** | Upload internal policy documents — employees self-serve answers without pinging HR |

The architecture is domain-agnostic — swap the documents, keep the pipeline.

---

## Stack

| Layer | Technology |
|---|---|
| API framework | [FastAPI](https://fastapi.tiangolo.com/) |
| RAG orchestration | [LangChain](https://python.langchain.com/) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (local, persistent) |
| Embeddings | `all-MiniLM-L6-v2` via [sentence-transformers](https://www.sbert.net/) (runs locally, no API key) |
| LLM | [Ollama](https://ollama.com/) (local, free) or [Claude](https://www.anthropic.com/) via Anthropic API — switchable via `.env` |

## How It Works

```
User uploads PDF / DOCX / TXT / MD
        │
        ▼
 Document Loader (LangChain)
        │
        ▼
 Text Splitter — chunks of 1000 chars, 200 overlap
        │
        ▼
 Embeddings — all-MiniLM-L6-v2 (runs locally, no API key)
        │
        ▼
 ChromaDB — persisted vector store

User asks a question
        │
        ▼
 Query embedded → top-k chunks retrieved from ChromaDB
        │
        ▼
 Retrieved chunks + question sent to LLM (Ollama or Claude)
        │
        ▼
 Grounded answer returned with source attribution
```

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload & ingest a document |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document |
| `POST` | `/query` | Ask a question |

Full interactive docs at `http://localhost:8000/docs` once running.

## Setup

### Prerequisites
- **Docker** (recommended) — or Python 3.11+ for local dev
- For Ollama: [install Ollama](https://ollama.com) and run `ollama pull llama3.2`
- For Claude: an [Anthropic API key](https://console.anthropic.com/)

---

### Option A — Docker (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/nimitzanalytica/rag-pipeline.git
cd rag-pipeline

# 2. Configure environment
cp .env.example .env
# Edit .env — set LLM_PROVIDER and add API key if using Claude

# 3. Build and start
docker compose up --build
```

The API will be live at `http://localhost:8000`.
Swagger UI at `http://localhost:8000/docs`.

To stop: `docker compose down`
Data persists in `chroma_db/` and `uploads/` across restarts.

---

### Option B — Local development (virtual environment)

```bash
# 1. Clone the repo
git clone https://github.com/nimitzanalytica/rag-pipeline.git
cd rag-pipeline

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set LLM_PROVIDER and add API key if using Claude

# 5. Start the server
uvicorn main:app --reload
```

The API will be live at `http://localhost:8000`.
Swagger UI at `http://localhost:8000/docs`.

---

### Frontend UI

Open `index.html` directly in your browser (no extra server needed) for a clean drag-and-drop interface to upload documents and ask questions.

---

## LLM Provider

Controlled by a single line in `.env`:

```dotenv
LLM_PROVIDER=ollama   # free, runs locally, no API key needed
LLM_PROVIDER=claude   # Anthropic API (requires ANTHROPIC_API_KEY)
```

**Ollama** is the default — ideal for privacy-sensitive environments where data must not leave your network.
**Claude** offers higher quality responses via the Anthropic API.

---

## Usage Examples

### Upload a document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "doc_id": "your_document_a1b2c3d4",
  "filename": "your_document.pdf",
  "chunks_created": 42,
  "pages": 10,
  "message": "Successfully ingested 'your_document.pdf' into 42 chunks."
}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main conclusions of this document?"}'
```

Response:
```json
{
  "answer": "Based on the document, the main conclusions are...",
  "sources": [
    {"filename": "your_document.pdf", "doc_id": "your_document_a1b2c3d4", "chunk_index": 3, "page": 1}
  ],
  "chunks_retrieved": 4,
  "provider": "ollama"
}
```

### Scope a query to one document

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize section 2", "doc_id": "your_document_a1b2c3d4"}'
```

### List all documents

```bash
curl http://localhost:8000/documents
```

### Delete a document

```bash
curl -X DELETE http://localhost:8000/documents/your_document_a1b2c3d4
```

---

## Project Structure

```
rag-pipeline/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── index.html            # Simple frontend UI (open directly in browser)
├── main.py               # FastAPI app — routes, request/response schemas
├── rag.py                # RAG pipeline — ingest, query, list, delete
└── config.py             # Settings via pydantic-settings + .env
```

---

## Configuration

All settings are controlled via `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `claude` |
| `ANTHROPIC_API_KEY` | *(required if using Claude)* | Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model to use |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model to use |
| `MAX_TOKENS` | `1024` | Max tokens in LLM response |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `4` | Chunks retrieved per query |

---

## Design Decisions

- **Swappable LLM** — switch between Ollama (local/free) and Claude (API) with one `.env` change. No code modifications required.
- **Local embeddings** — `sentence-transformers` runs entirely on your machine regardless of LLM provider. No OpenAI key needed.
- **Persistent ChromaDB** — the vector store survives server restarts. Documents stay ingested until explicitly deleted.
- **Source attribution** — every answer includes the filenames and chunk indices used, making answers traceable and auditable.
- **Duplicate detection** — re-uploading the same file returns a `409 Conflict` rather than creating duplicate embeddings.
- **CORS enabled** — the API accepts requests from the local `index.html` frontend opened as a `file://` URL.

## License

MIT
