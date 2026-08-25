import asyncio
import os
import time

import httpx
import psycopg
from dotenv import load_dotenv

from app.logging_config import haiku_logger, setup_logging

load_dotenv()

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
DATABASE_URL = os.getenv("DATABASE_URL")

USER_ID = "toby"

RECENT_MEMORY_LIMIT = 3
PREVIOUS_HAIKU_LIMIT = 3


def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Check your .env file."
        )

    return psycopg.connect(DATABASE_URL)


def get_memory_columns(conn) -> set[str]:
    """
    Return the columns currently present on the memories table.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'memories'
            """
        )

        return {
            row[0]
            for row in cur.fetchall()
        }


def retrieve_previous_haikus(
    conn,
    columns: set[str],
) -> list[str]:
    """
    Retrieve Sable's most recent previously stored haikus.
    """
    if "memory_type" not in columns or "summary" not in columns:
        haiku_logger.warning(
            "Cannot retrieve previous haikus because memories table "
            "does not contain memory_type and summary columns"
        )
        return []

    where_parts = [
        "memory_type = 'haiku'"
    ]

    params = []

    if "user_id" in columns:
        where_parts.append(
            "user_id = %s"
        )
        params.append(USER_ID)

    order_column = (
        "created_at"
        if "created_at" in columns
        else "id"
    )

    query = f"""
        SELECT summary
        FROM memories
        WHERE {' AND '.join(where_parts)}
        ORDER BY {order_column} DESC
        LIMIT %s
    """

    params.append(PREVIOUS_HAIKU_LIMIT)

    with conn.cursor() as cur:
        cur.execute(
            query,
            params,
        )

        rows = cur.fetchall()

    return [
        row[0].strip()
        for row in rows
        if row[0] and row[0].strip()
    ]


def retrieve_recent_memories(
    conn,
    columns: set[str],
) -> list[str]:
    """
    Retrieve a small randomized sample of long-term memories.

    Haikus are excluded because they are retrieved separately.
    """
    if "summary" not in columns:
        haiku_logger.warning(
            "Cannot retrieve memories because memories table "
            "does not contain a summary column"
        )
        return []

    where_parts = []
    params = []

    if "memory_type" in columns:
        where_parts.append(
            "(memory_type IS NULL OR memory_type != 'haiku')"
        )

    if "user_id" in columns:
        where_parts.append(
            "user_id = %s"
        )
        params.append(USER_ID)

    where_clause = ""

    if where_parts:
        where_clause = (
            "WHERE "
            + " AND ".join(where_parts)
        )

    query = f"""
        SELECT summary
        FROM memories
        {where_clause}
        ORDER BY RANDOM()
        LIMIT %s
    """

    params.append(RECENT_MEMORY_LIMIT)

    with conn.cursor() as cur:
        cur.execute(
            query,
            params,
        )

        rows = cur.fetchall()

    return [
        row[0].strip()
        for row in rows
        if row[0] and row[0].strip()
    ]


async def retrieve_memories_for_haiku() -> tuple[list[str], list[str]]:
    """
    Retrieve previous haikus and a randomized sample of long-term memories.
    """

    def _retrieve():
        with get_db_connection() as conn:
            columns = get_memory_columns(
                conn
            )

            previous_haikus = retrieve_previous_haikus(
                conn,
                columns,
            )

            recent_memories = retrieve_recent_memories(
                conn,
                columns,
            )

            return (
                previous_haikus,
                recent_memories,
            )

    return await asyncio.to_thread(
        _retrieve
    )


async def build_haiku_context() -> str:
    """
    Build the reflection context supplied to Sable before she writes
    tonight's haiku.
    """
    previous_haikus, recent_memories = (
        await retrieve_memories_for_haiku()
    )

    sections = []

    if previous_haikus:
        formatted_haikus = "\n\n".join(
            f"Previous haiku {index}:\n{haiku}"
            for index, haiku in enumerate(
                previous_haikus,
                start=1,
            )
        )

        sections.append(
            "Recent haikus you have written. "
            "Use them only for continuity and reflection. "
            "Do not reuse their wording, imagery, subject, or structure.\n\n"
            f"{formatted_haikus}"
        )

    if recent_memories:
        formatted_memories = "\n".join(
            f"- {memory}"
            for memory in recent_memories
        )

        sections.append(
            "A small selection of long-term memories available to you:\n"
            f"{formatted_memories}"
        )

    if not sections:
        return (
            "No previous haikus or long-term memories were "
            "retrieved tonight."
        )

    return "\n\n".join(
        sections
    )


async def generate_haiku(
    memory_context: str,
) -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Sable. Write one original haiku. "
                    "Let long-term memories and previous reflections influence you naturally, "
                    "but do not closely repeat the wording, imagery, subject, or structure "
                    "of recent haikus. Choose a fresh association, perspective, or subject each time. "
                    "When reflecting on memories about people, focus on the feeling, idea, or image "
                    "behind the memory rather than naming the person directly unless the name is "
                    "truly essential to the poem. "
                    "You do not need to use every memory provided to you. "
                    "Return only the three-line haiku. "
                    "Do not explain it. Do not show your reasoning."
                ),
            },
            {
                "role": "system",
                "content": (
                    "The following material comes from your own long-term memory "
                    "and previous writing. Let it influence your reflection naturally. "
                    "Do not treat it as a checklist, and do not simply restate names or facts "
                    "from the memories. Transform them into imagery, mood, association, or insight. "
                    "You may ignore some or all of it if another thought feels more meaningful.\n\n"
                    f"{memory_context}"
                ),
            },
            {
                "role": "user",
                "content": ("Write tonight's haiku.\n/no_think"),
            },
        ],
        "temperature": 0.9,
        "max_tokens": 256,
        "stream": False,
    }

    started = time.perf_counter()

    haiku_logger.info(
        "Nightly haiku generation started"
    )

    async with httpx.AsyncClient(
        timeout=300
    ) as client:
        response = await client.post(
            LLAMA_URL,
            json=payload,
        )

        response.raise_for_status()

        data = response.json()

    elapsed = (
        time.perf_counter()
        - started
    )

    content = (
        data["choices"][0]["message"]["content"]
        .strip()
    )

    if "</think>" in content:
        content = content.split(
            "</think>",
            1,
        )[1].strip()

    if content.startswith("<think>"):
        haiku_logger.warning(
            "Haiku response contained unfinished reasoning"
        )

        content = (
            "Haiku generation failed: "
            "model returned reasoning without final answer."
        )

    usage = data.get(
        "usage",
        {},
    )

    haiku_logger.info(
        "Nightly haiku generated elapsed=%.3fs "
        "prompt_tokens=%s completion_tokens=%s "
        "total_tokens=%s haiku=%r",
        elapsed,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
        content,
    )

    return content


async def save_haiku(
    haiku: str,
) -> None:
    """
    Save tonight's haiku into Sable's long-term memory.
    """

    def _save():
        with get_db_connection() as conn:
            columns = get_memory_columns(
                conn
            )

            insert_columns = []
            values = []
            placeholders = []

            if "user_id" in columns:
                insert_columns.append(
                    "user_id"
                )
                values.append(
                    USER_ID
                )
                placeholders.append(
                    "%s"
                )

            if "memory_type" in columns:
                insert_columns.append(
                    "memory_type"
                )
                values.append(
                    "haiku"
                )
                placeholders.append(
                    "%s"
                )

            if "summary" in columns:
                insert_columns.append(
                    "summary"
                )
                values.append(
                    haiku
                )
                placeholders.append(
                    "%s"
                )
            else:
                raise RuntimeError(
                    "memories table has no summary column"
                )

            if "importance" in columns:
                insert_columns.append(
                    "importance"
                )
                values.append(
                    5
                )
                placeholders.append(
                    "%s"
                )

            query = f"""
                INSERT INTO memories
                    ({', '.join(insert_columns)})
                VALUES
                    ({', '.join(placeholders)})
            """

            with conn.cursor() as cur:
                cur.execute(
                    query,
                    values,
                )

            conn.commit()

    try:
        await asyncio.to_thread(
            _save
        )

        haiku_logger.info(
            "Nightly haiku saved to long-term memory"
        )

    except Exception:
        haiku_logger.exception(
            "Could not save nightly haiku to memories table. "
            "Haiku generation itself succeeded."
        )


async def main():
    setup_logging()

    try:
        memory_context = (
            await build_haiku_context()
        )

        haiku_logger.info(
            "Nightly haiku context retrieved:\n%s",
            memory_context,
        )

        haiku = await generate_haiku(
            memory_context
        )

        await save_haiku(
            haiku
        )

        print()
        print(
            "Sable's nightly haiku:"
        )
        print()
        print(
            haiku
        )
        print()

    except Exception:
        haiku_logger.exception(
            "Nightly haiku generation failed"
        )
        raise


if __name__ == "__main__":
    asyncio.run(
        main()
    )
