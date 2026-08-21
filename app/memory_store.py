import json
from typing import Optional

import psycopg


class MemoryStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def store_memory(
        self,
        memory_type: str,
        summary: str,
        embedding: list[float],
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_text: Optional[str] = None,
        importance: int = 5,
        confidence: float = 1.0,
        source_message_id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Store a long-term memory and return its database ID.
        """

        if metadata is None:
            metadata = {}

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memories (
                        memory_type,
                        subject,
                        predicate,
                        object_text,
                        summary,
                        importance,
                        confidence,
                        embedding,
                        metadata
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s::jsonb
                    )
                    RETURNING id
                    """,
                    (
                        memory_type,
                        subject,
                        predicate,
                        object_text,
                        summary,
                        importance,
                        confidence,
                        embedding,
                        json.dumps(metadata),
                    ),
                )

                memory_id = cur.fetchone()[0]

                if source_message_id is not None:
                    cur.execute(
                        """
                        INSERT INTO memory_sources (
                            memory_id,
                            message_id
                        )
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            memory_id,
                            source_message_id,
                        ),
                    )

            conn.commit()

        return memory_id
