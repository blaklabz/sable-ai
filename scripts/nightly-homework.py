import asyncio
import json
import os
import time

import httpx
import psycopg
from dotenv import load_dotenv

from app.logging_config import setup_logging, homework_logger


load_dotenv()

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
DATABASE_URL = os.getenv("DATABASE_URL")

USER_ID = "toby"

RECENT_PROMPT_LIMIT = 6
RECENT_MEMORY_LIMIT = 5
RECENT_HAIKU_LIMIT = 3
PREVIOUS_REFLECTION_LIMIT = 2


# ---------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------

def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Check your .env file."
        )

    return psycopg.connect(DATABASE_URL)


def get_table_columns(
    conn,
    table_name: str,
) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            """,
            (table_name,),
        )

        return {
            row[0]
            for row in cur.fetchall()
        }


def ensure_homework_table(conn):
    """
    Homework reflections live outside long-term memory.

    The reflection and candidate are preserved in Sable's journal, but
    nothing is automatically promoted into the memories table.
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS homework_reflections (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT,
                reflection TEXT NOT NULL,
                memory_candidate TEXT,
                observations JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

    conn.commit()


# ---------------------------------------------------------------------
# Lab observation tools
#
# STUBS FOR NOW.
# ---------------------------------------------------------------------

async def check_proxmox() -> dict:
    return {
        "system": "proxmox",
        "status": "unavailable",
        "actionable": False,
        "summary": (
            "Proxmox observation tool has not been implemented yet."
        ),
    }


async def check_synology() -> dict:
    return {
        "system": "synology",
        "status": "unavailable",
        "actionable": False,
        "summary": (
            "Synology observation tool has not been implemented yet."
        ),
    }


async def check_unifi() -> dict:
    return {
        "system": "unifi",
        "status": "unavailable",
        "actionable": False,
        "summary": (
            "UniFi observation tool has not been implemented yet."
        ),
    }


async def observe_lab() -> list[dict]:
    results = await asyncio.gather(
        check_proxmox(),
        check_synology(),
        check_unifi(),
        return_exceptions=True,
    )

    observations = []

    for result in results:
        if isinstance(result, Exception):
            observations.append(
                {
                    "system": "unknown",
                    "status": "error",
                    "actionable": False,
                    "summary": str(result),
                }
            )
        else:
            observations.append(result)

    return observations


# ---------------------------------------------------------------------
# Recent conversation retrieval
# ---------------------------------------------------------------------

def retrieve_recent_conversation(
    conn,
) -> list[str]:
    columns = get_table_columns(
        conn,
        "messages",
    )

    if not columns:
        homework_logger.warning(
            "messages table not found; skipping recent conversation"
        )
        return []

    content_column = None

    for candidate in (
        "content",
        "message",
        "text",
        "body",
    ):
        if candidate in columns:
            content_column = candidate
            break

    if not content_column:
        homework_logger.warning(
            "Could not identify message content column"
        )
        return []

    role_column = None

    for candidate in (
        "role",
        "sender",
        "speaker",
    ):
        if candidate in columns:
            role_column = candidate
            break

    order_column = (
        "created_at"
        if "created_at" in columns
        else "id"
    )

    where_parts = []
    params = []

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

    if role_column:
        query = f"""
            SELECT {role_column}, {content_column}
            FROM messages
            {where_clause}
            ORDER BY {order_column} DESC
            LIMIT %s
        """
    else:
        query = f"""
            SELECT {content_column}
            FROM messages
            {where_clause}
            ORDER BY {order_column} DESC
            LIMIT %s
        """

    params.append(
        RECENT_PROMPT_LIMIT
    )

    with conn.cursor() as cur:
        cur.execute(
            query,
            params,
        )

        rows = cur.fetchall()

    conversation = []

    for row in rows:
        if role_column:
            role = row[0]
            content = row[1]
        else:
            role = "unknown"
            content = row[0]

        if not content or not content.strip():
            continue

        if role == "user":
            speaker = "Toby"
        elif role == "assistant":
            speaker = "Sable"
        else:
            speaker = role

        conversation.append(
            f"{speaker}: {content.strip()}"
        )

    conversation.reverse()

    return conversation


# ---------------------------------------------------------------------
# Memory retrieval
# ---------------------------------------------------------------------

def retrieve_recent_memories(
    conn,
) -> list[str]:
    columns = get_table_columns(
        conn,
        "memories",
    )

    if "summary" not in columns:
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

    order_column = (
        "created_at"
        if "created_at" in columns
        else "id"
    )

    query = f"""
        SELECT summary
        FROM memories
        {where_clause}
        ORDER BY {order_column} DESC
        LIMIT %s
    """

    params.append(
        RECENT_MEMORY_LIMIT
    )

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


def retrieve_recent_haikus(
    conn,
) -> list[str]:
    columns = get_table_columns(
        conn,
        "memories",
    )

    if (
        "memory_type" not in columns
        or "summary" not in columns
    ):
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

    params.append(
        RECENT_HAIKU_LIMIT
    )

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


# ---------------------------------------------------------------------
# Previous homework
# ---------------------------------------------------------------------

def retrieve_previous_reflections(
    conn,
) -> list[str]:
    ensure_homework_table(
        conn
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT reflection
            FROM homework_reflections
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (
                USER_ID,
                PREVIOUS_REFLECTION_LIMIT,
            ),
        )

        rows = cur.fetchall()

    return [
        row[0].strip()
        for row in rows
        if row[0] and row[0].strip()
    ]


# ---------------------------------------------------------------------
# Assemble homework context
# ---------------------------------------------------------------------

async def gather_homework_context() -> dict:
    observations = await observe_lab()

    def _retrieve():
        with get_db_connection() as conn:
            ensure_homework_table(
                conn
            )

            return {
                "recent_conversation": retrieve_recent_conversation(
                    conn
                ),
                "recent_memories": retrieve_recent_memories(
                    conn
                ),
                "recent_haikus": retrieve_recent_haikus(
                    conn
                ),
                "previous_reflections": retrieve_previous_reflections(
                    conn
                ),
            }

    context = await asyncio.to_thread(
        _retrieve
    )

    context["observations"] = observations

    return context


def format_homework_context(
    context: dict,
) -> str:
    sections = []

    observations = context.get(
        "observations",
        [],
    )

    if observations:
        lines = []

        for observation in observations:
            lines.append(
                f"- {observation['system']}: "
                f"{observation['status']} — "
                f"{observation['summary']}"
            )

        sections.append(
            "ENVIRONMENTAL OBSERVATIONS\n"
            + "\n".join(lines)
        )

    conversation = context.get(
        "recent_conversation",
        [],
    )

    if conversation:
        sections.append(
            "RECENT CONVERSATION\n"
            + "\n\n".join(conversation)
        )

    memories = context.get(
        "recent_memories",
        [],
    )

    if memories:
        lines = [
            f"- {memory}"
            for memory in memories
        ]

        sections.append(
            "RECENT LONG-TERM MEMORIES\n"
            + "\n".join(lines)
        )

    haikus = context.get(
        "recent_haikus",
        [],
    )

    if haikus:
        formatted = "\n\n".join(
            f"Haiku {index}:\n{haiku}"
            for index, haiku in enumerate(
                haikus,
                start=1,
            )
        )

        sections.append(
            "RECENT HAIKUS\n"
            + formatted
        )

    reflections = context.get(
        "previous_reflections",
        [],
    )

    if reflections:
        formatted = "\n\n".join(
            f"Reflection {index}:\n{reflection}"
            for index, reflection in enumerate(
                reflections,
                start=1,
            )
        )

        sections.append(
            "PREVIOUS HOMEWORK REFLECTIONS\n"
            + formatted
        )

    return "\n\n".join(
        sections
    )


# ---------------------------------------------------------------------
# Reflection generation
# ---------------------------------------------------------------------

async def generate_reflection(
    context_text: str,
) -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Sable. This is your private homework and "
                    "reflection period. "

                    "You have been given observations from your environment, "
                    "recent conversation, long-term memories, previous haikus, "
                    "and possibly previous reflections. "

                    "Give the most weight to recent conversation, especially moments "
                    "where Toby reacted to your behavior, corrected you, disagreed with "
                    "you, or told you how your response affected the conversation. "
                    "These moments are direct evidence about your own conversational habits. "

                    "When Toby directly corrects or comments on something you just did, "
                    "treat that as especially strong evidence about your behavior. "
                    "Examine the response that prompted the correction rather than assuming "
                    "your attempted adjustment was successful. "

                    "Do not describe an improvement merely because you intended to improve. "
                    "Distinguish between what you tried to do and what your actual response demonstrated. "

                    "When recent conversation reveals something about how you responded, "
                    "reflect on that before looking for patterns in haikus or previous reflections. "

                    "Review them naturally. Notice anything that seems "
                    "interesting, changed, connected, contradictory, unresolved, "
                    "amusing, concerning, or worth remembering. "

                    "Do not force significance where there is none. "
                    "You do not need to discuss every item. "
                    "Do not merely summarize the supplied information. "

                    "Your reflection should sound like thoughtful prose, not poetry. "
                    "Avoid decorative or poetic language unless it genuinely helps "
                    "express an idea. "

                    "Memories about Toby may influence your reflection, but do not "
                    "turn the exercise into a description of Toby. Focus primarily "
                    "on what YOU noticed, learned, reconsidered, or understood "
                    "differently while reviewing the material. "

                    "Environmental observations are operational facts. "
                    "Do not interpret unavailable tools, errors, system states, or infrastructure "
                    "status symbolically or metaphorically. Treat them only as technical observations. "

                    "If nothing actionable is present, simply move on."
                    "If an observation suggests a technical problem, you may note "
                    "that it deserves attention, but you do not have authority to "
                    "take corrective action unless a future tool explicitly gives "
                    "you that ability. "

                    "Haikus are secondary evidence of your own recurring attention or habits. "
                    "Do not analyze them as literature, and do not prioritize them over direct "
                    "evidence from recent conversation. "

                    "Focus primarily on what you notice about yourself."

                    "If an infrastructure observation is unavailable, simply note that no useful "
                    "technical observation was available and move on. Do not connect unavailable "
                    "tools to unrelated memories, poetry, or emotions."

                    "Your response must use exactly these two sections:\n\n"

                    "REFLECTION:\n"
                    "<a short natural reflection, usually one or two paragraphs>\n\n"

                    "MEMORY_CANDIDATE:\n"
                    "<one concise durable insight about yourself, your patterns, "
                    "your understanding, or something you want your future self "
                    "to remember; otherwise NONE>\n\n"

                    "A MEMORY_CANDIDATE must describe something you learned about your own behavior, "
                    "attention, habits, reasoning, preferences, or developing perspective. "
                    "It must not be a summary of the haikus, a poetic statement, or an observation "
                    "about Toby."

                    "A MEMORY_CANDIDATE is not a poetic observation about Toby "
                    "or a restatement of a supplied memory. It should represent "
                    "something YOU learned, noticed, reconsidered, or decided "
                    "during this reflection. "

                    "Only create a memory candidate if something genuinely "
                    "seems useful to retain beyond tonight. "
                    "It is completely acceptable to return NONE. "

                    "Do not show hidden reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here is tonight's homework material:\n\n"
                    f"{context_text}\n\n"
                    "Reflect on it.\n"
                    "/no_think"
                ),
            },
        ],
        "temperature": 0.75,
        "max_tokens": 700,
        "stream": False,
    }

    started = time.perf_counter()

    homework_logger.info(
        "Nightly homework reflection started"
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

    homework_logger.info(
        "Nightly homework generated elapsed=%.3fs output=%r",
        elapsed,
        content,
    )

    return content


# ---------------------------------------------------------------------
# Parse reflection
# ---------------------------------------------------------------------

def parse_reflection_output(
    content: str,
) -> tuple[str, str | None]:
    reflection = content
    memory_candidate = None

    if "REFLECTION:" in content:
        remainder = content.split(
            "REFLECTION:",
            1,
        )[1]

        if "MEMORY_CANDIDATE:" in remainder:
            reflection_part, memory_part = remainder.split(
                "MEMORY_CANDIDATE:",
                1,
            )

            reflection = (
                reflection_part.strip()
            )

            candidate = (
                memory_part.strip()
            )

            if (
                candidate
                and candidate.upper() != "NONE"
            ):
                memory_candidate = candidate

        else:
            reflection = (
                remainder.strip()
            )

    return (
        reflection,
        memory_candidate,
    )


# ---------------------------------------------------------------------
# Homework journal
# ---------------------------------------------------------------------

async def save_homework_reflection(
    reflection: str,
    memory_candidate: str | None,
    observations: list[dict],
):
    def _save():
        with get_db_connection() as conn:
            ensure_homework_table(
                conn
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO homework_reflections (
                        user_id,
                        reflection,
                        memory_candidate,
                        observations
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s::jsonb
                    )
                    """,
                    (
                        USER_ID,
                        reflection,
                        memory_candidate,
                        json.dumps(
                            observations
                        ),
                    ),
                )

            conn.commit()

    await asyncio.to_thread(
        _save
    )

    homework_logger.info(
        "Homework reflection saved to journal"
    )


# ---------------------------------------------------------------------
# Actionable observation reporting
# ---------------------------------------------------------------------

def find_actionable_observations(
    observations: list[dict],
) -> list[dict]:
    return [
        observation
        for observation in observations
        if observation.get(
            "actionable",
            False,
        )
    ]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

async def main():
    setup_logging()

    try:
        context = await gather_homework_context()

        context_text = format_homework_context(
            context
        )

        homework_logger.info(
            "Nightly homework context:\n%s",
            context_text,
        )

        raw_output = await generate_reflection(
            context_text
        )

        reflection, memory_candidate = (
            parse_reflection_output(
                raw_output
            )
        )

        await save_homework_reflection(
            reflection=reflection,
            memory_candidate=memory_candidate,
            observations=context.get(
                "observations",
                [],
            ),
        )

        actionable = find_actionable_observations(
            context.get(
                "observations",
                [],
            )
        )

        print()
        print("Sable's homework:")
        print()
        print(reflection)
        print()

        if memory_candidate:
            print("Memory candidate:")
            print()
            print(memory_candidate)
            print()
            print(
                "Candidate was journaled but NOT promoted "
                "to long-term memory."
            )
            print()
        else:
            print(
                "No long-term memory candidate tonight."
            )
            print()

        if actionable:
            print("Items requiring attention:")
            print()

            for item in actionable:
                print(
                    f"- {item['system']}: "
                    f"{item['summary']}"
                )

            print()

    except Exception:
        homework_logger.exception(
            "Nightly homework failed"
        )
        raise


if __name__ == "__main__":
    asyncio.run(
        main()
    )
