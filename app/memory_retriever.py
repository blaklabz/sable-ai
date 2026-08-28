import os
from typing import Any

from sentence_transformers import SentenceTransformer

from app.db import get_connection


EMBEDDING_MODEL = os.getenv(
    "SABLE_EMBEDDING_MODEL",
    "/home/nixy/models/embeddings/bge-small-en-v1.5",
)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()

    embedding = model.encode(
        text,
        normalize_embeddings=True,
    )

    return [float(value) for value in embedding]


def normalize_query(query: str) -> str:
    cleaned = query.strip()

    prefixes = (
        "sable,",
        "sable ",
        "hey sable,",
        "hey sable ",
    )

    lowered = cleaned.lower()

    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    lowered = cleaned.lower()

    if "what bikes do i have" in lowered:
        return "What bikes does Toby have?"

    return cleaned


def detect_requested_memory_type(query: str) -> str | None:
    lowered = query.lower()

    if any(
        phrase in lowered
        for phrase in (
            "haiku",
            "haikus",
            "poem",
            "poems",
        )
    ):
        return "haiku"

    return None


def retrieve_memories_by_type(
    memory_type: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    sql = """
        SELECT
            id,
            memory_type,
            summary,
            importance,
            confidence,
            created_at
        FROM memories
        WHERE status = 'active'
          AND memory_type = %s
        ORDER BY created_at DESC
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    memory_type,
                    limit,
                ),
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "memory_type": row[1],
            "summary": row[2],
            "importance": row[3],
            "confidence": row[4],
            "created_at": row[5],
            "similarity": None,
        }
        for row in rows
    ]


def retrieve_semantic_memories(
    query: str,
    limit: int = 3,
    min_similarity: float = 0.55,
) -> list[dict[str, Any]]:
    query_embedding = embed_text(normalize_query(query))

    vector = "[" + ",".join(str(value) for value in query_embedding) + "]"

    sql = """
        SELECT
            id,
            memory_type,
            summary,
            importance,
            confidence,
            1 - (embedding <=> %s::vector) AS similarity
        FROM memories
        WHERE status = 'active'
          AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    vector,
                    vector,
                    limit,
                ),
            )
            rows = cur.fetchall()

    memories = [
        {
            "id": row[0],
            "memory_type": row[1],
            "summary": row[2],
            "importance": row[3],
            "confidence": row[4],
            "similarity": float(row[5]),
        }
        for row in rows
    ]

    return [
        memory
        for memory in memories
        if memory["similarity"] >= min_similarity
    ]


def retrieve_memories(
    query: str,
    limit: int = 3,
    min_similarity: float = 0.55,
) -> list[dict[str, Any]]:
    requested_type = detect_requested_memory_type(query)

    if requested_type:
        return retrieve_memories_by_type(
            requested_type,
            limit=5,
        )

    return retrieve_semantic_memories(
        query,
        limit=limit,
        min_similarity=min_similarity,
    )
