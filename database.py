"""
database.py — PostgreSQL query logging

Every query submitted to the API is logged to the query_logs table.
This enables auditing, analytics, and system improvement over time.

Table schema:
    id            — auto-incrementing primary key
    question      — the user's question
    answer        — the LLM's response
    doc_id        — document scoped to (null = all documents)
    chunks_retrieved — number of chunks used to answer
    provider      — llm provider used (ollama or claude)
    sources       — filenames of source documents (JSON array)
    created_at    — timestamp of the query
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor

from config import settings

logger = logging.getLogger(__name__)


def get_connection():
    """Open a new PostgreSQL connection."""
    return psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )


@contextmanager
def db_cursor():
    """Context manager — opens a connection, yields a cursor, commits, closes."""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def init_db():
    """
    Create the query_logs table if it doesn't exist.
    Called once at application startup.
    """
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS query_logs (
            id               SERIAL PRIMARY KEY,
            question         TEXT NOT NULL,
            answer           TEXT NOT NULL,
            doc_id           TEXT,
            chunks_retrieved INTEGER NOT NULL DEFAULT 0,
            provider         TEXT NOT NULL,
            sources          JSONB,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """
    try:
        with db_cursor() as cur:
            cur.execute(create_table_sql)
        logger.info("Database initialized — query_logs table ready")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}. Query logging will be disabled.")


def log_query(
    question: str,
    answer: str,
    provider: str,
    chunks_retrieved: int,
    sources: list[dict],
    doc_id: str | None = None,
) -> None:
    """
    Insert a query log record into PostgreSQL.
    Failures are caught and logged — they never break the API response.
    """
    insert_sql = """
        INSERT INTO query_logs
            (question, answer, doc_id, chunks_retrieved, provider, sources, created_at)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with db_cursor() as cur:
            cur.execute(insert_sql, (
                question,
                answer,
                doc_id,
                chunks_retrieved,
                provider,
                json.dumps(sources),
                datetime.now(timezone.utc),
            ))
    except Exception as e:
        # Logging failure must never break the API — just warn and continue
        logger.warning(f"Failed to log query to database: {e}")
