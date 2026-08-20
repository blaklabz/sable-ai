from sentence_transformers import SentenceTransformer

from app.db import get_connection


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_model = None


def get_embedding_model():
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

    return embedding.tolist()


def retrieve_memories(
    query: str,
    limit: int = 5,
) -> list[dict]:
    query_embedding = embed_text(query)

    # pgvector accepts the textual representation:
    # [0.123,-0.456,...]
    vector = "[" + ",".join(str(x) for x in query_embedding) + "]"

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

    return [
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
