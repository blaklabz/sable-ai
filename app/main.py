import json
import os
import time
from pathlib import Path

import httpx
import psycopg
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.logging_config import (
    api_logger,
    llm_logger,
    memory_logger,
    prompt_logger,
    setup_logging,
)
from app.memory_retriever import embed_text, retrieve_memories
from app.memory_store import MemoryStore

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"

PROMPT_FILES = [
    "system-prompt.txt",
    "identity.txt",
    "purpose.txt",
    "values.txt",
    "personality.txt",
    "communication.txt",
    "engineering.txt",
    "reasoning.txt",
    "humor.txt",
    "relationship.txt",
    "memory-rules.txt",
]

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

DATABASE_URL = os.getenv("DATABASE_URL")

USER_ID = "toby"
CONVERSATION_ID = 1


setup_logging()

app = FastAPI(
    title="Sable"
)

templates = Jinja2Templates(
    directory=str(
        BASE_DIR
        / "app"
        / "templates"
    )
)

memory_store = MemoryStore()


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = Field(
        default_factory=list
    )


# ---------------------------------------------------------------------
# Database / message history
# ---------------------------------------------------------------------

def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Check your .env file."
        )

    return psycopg.connect(
        DATABASE_URL
    )


def save_message(
    conversation_id: int,
    role: str,
    content: str,
) -> int:
    """
    Save one visible conversation message.

    Both Toby's messages and Sable's visible replies are preserved so
    later systems such as nightly homework can review recent interactions.
    """

    content = content.strip()

    if not content:
        raise ValueError(
            "Cannot save an empty message"
        )

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (
                    conversation_id,
                    role,
                    content
                )
                VALUES (
                    %s,
                    %s,
                    %s
                )
                RETURNING id
                """,
                (
                    conversation_id,
                    role,
                    content,
                ),
            )

            row = cur.fetchone()

            if not row:
                raise RuntimeError(
                    "Message insert did not return an id"
                )

            message_id = row[0]

        conn.commit()

    api_logger.info(
        "Message saved id=%s "
        "conversation_id=%s role=%s chars=%d",
        message_id,
        conversation_id,
        role,
        len(content),
    )

    return message_id


# ---------------------------------------------------------------------
# LLM response cleanup
# ---------------------------------------------------------------------

def clean_llm_reply(
    content: str,
) -> str:
    """
    Remove Qwen reasoning blocks from the visible reply before returning
    it to the UI or saving it into conversation history.
    """

    content = content.strip()

    if "</think>" in content:
        content = content.split(
            "</think>",
            1,
        )[1].strip()

    if content.startswith("<think>"):
        llm_logger.warning(
            "Main response contained unfinished reasoning"
        )

        return (
            "I got tangled up in my internal reasoning "
            "and didn't produce a clean response."
        )

    return content


# ---------------------------------------------------------------------
# Web interface
# ---------------------------------------------------------------------

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(
    request: Request,
):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "user_id": USER_ID
        },
    )


# ---------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------

def load_system_prompt() -> str:
    sections = []
    loaded_files = []

    for filename in PROMPT_FILES:
        prompt_file = (
            CONFIG_DIR
            / filename
        )

        try:
            content = (
                prompt_file
                .read_text(
                    encoding="utf-8"
                )
                .strip()
            )

        except Exception:
            prompt_logger.exception(
                "Failed loading prompt file=%s",
                filename,
            )
            raise

        if content:
            section_name = (
                prompt_file
                .stem
                .upper()
            )

            sections.append(
                f"## {section_name}\n"
                f"{content}"
            )

            loaded_files.append(
                filename
            )

    system_prompt = (
        "\n\n".join(
            sections
        )
    )

    prompt_logger.info(
        "System prompt loaded "
        "files=%d chars=%d file_names=%s",
        len(loaded_files),
        len(system_prompt),
        ",".join(
            loaded_files
        ),
    )

    return system_prompt


# ---------------------------------------------------------------------
# Memory context
# ---------------------------------------------------------------------

def build_memory_context(
    memories: list[dict],
) -> str:
    if not memories:
        memory_logger.info(
            "No memories injected into prompt"
        )
        return ""

    lines = [
        "## RELEVANT LONG-TERM MEMORY",
        (
            "The following entries are retrieved records from your "
            "long-term memory. Treat them as available memories, not "
            "as guesses or vague impressions."
        ),
    ]

    for memory in memories:
        memory_logger.info(
            "Injecting memory "
            "id=%s type=%s similarity=%s "
            "importance=%s summary=%r",
            memory.get("id"),
            memory.get("memory_type"),
            memory.get("similarity"),
            memory.get("importance"),
            memory.get("summary"),
        )

        memory_type = (
            memory.get("memory_type")
            or "unknown"
        )

        lines.append(
            f"- [{memory_type}] "
            f"{memory['summary']}"
        )

    memory_context = (
        "\n".join(
            lines
        )
    )

    memory_logger.info(
        "Memory context built "
        "memories=%d chars=%d ids=%s",
        len(memories),
        len(memory_context),
        ",".join(
            str(
                memory.get(
                    "id",
                    "unknown",
                )
            )
            for memory in memories
        ),
    )

    return memory_context


# ---------------------------------------------------------------------
# Memory extraction
# ---------------------------------------------------------------------

async def extract_memory(
    user_message: str,
) -> dict | None:
    prompt = f"""
You are Sable's long-term memory evaluator.

Decide whether the following message from Toby contains information that
would be useful to remember in future conversations.

Only remember information that is likely to matter in future conversations, such as:
- preferences
- possessions
- ongoing projects
- important plans
- stable facts about Toby
- relationships
- recurring habits
- important personal context
- meaningful personal stories and experiences
- durable preferences about how Toby wants Sable to communicate or interact with him

Meaningful personal stories and experiences should usually become episodic memories.

A personal story is worth remembering when it reveals something lasting about Toby, such as:
- how he became interested in something
- an experience that shaped how he thinks
- an important childhood or family experience
- a meaningful success, failure, discovery, or lesson
- an experience that helps explain his personality, interests, values, or behavior
- a story involving an important relationship or period of his life

A memory does not need to be currently actionable to be worth remembering.
Some experiences matter because they are part of Toby's personal history.

For example:

Toby says:
"When I was a kid I threw my toy planes out of a second-floor window because I thought they would glide. They broke, so that weekend I went to the library to learn what made planes fly."

This SHOULD be remembered as an episodic memory because it is a meaningful childhood story that demonstrates Toby's curiosity and how an experiment led him to learn about flight.

A suitable memory would be:

{{
  "remember": true,
  "memory_type": "episodic",
  "subject": "Toby",
  "predicate": "childhood_experience",
  "object_text": "learning how airplanes fly",
  "summary": "As a child, Toby threw his toy planes from a second-floor window expecting them to glide. After they broke, he went to the library to learn what makes airplanes fly.",
  "importance": 7,
  "confidence": 1.0
}}

Do NOT remember:
- ordinary questions
- temporary status
- casual greetings
- one-time commands or requests that only apply to the current interaction
- routine daily events with no lasting significance
- meals or activities that are only relevant today
- information that is only relevant to the current conversation
- facts invented or inferred beyond what Toby explicitly said

Do not create an episodic memory merely because Toby described something that happened.
An event should have some lasting personal, historical, emotional, relational, or explanatory value.

Allowed memory_type values:
- semantic
- episodic
- preference
- self
- relationship
- plan

If nothing should be remembered, return exactly:

{{"remember": false}}

If something should be remembered, return JSON like:

{{
  "remember": true,
  "memory_type": "preference",
  "subject": "Toby",
  "predicate": "likes",
  "object_text": "Peanut Chews",
  "summary": "Toby likes Peanut Chews.",
  "importance": 6,
  "confidence": 1.0
}}

Importance must be an integer from 1 through 10.
Confidence must be between 0.0 and 1.0.

Return JSON only.
Do not include markdown.
Do not include commentary.
Do not explain your decision.

Toby's message:
{user_message}
"""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract structured "
                    "long-term memories."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.1,
        "max_tokens": 1536,
        "stream": False,
    }

    llm_started = (
        time.perf_counter()
    )

    llm_logger.info(
        "Memory extraction started "
        "message_chars=%d prompt_chars=%d",
        len(user_message),
        len(prompt),
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

    llm_elapsed = (
        time.perf_counter()
        - llm_started
    )

    usage = data.get(
        "usage",
        {},
    )

    llm_logger.info(
        "Memory extraction completed "
        "elapsed=%.3fs "
        "prompt_tokens=%s "
        "completion_tokens=%s "
        "total_tokens=%s",
        llm_elapsed,
        usage.get(
            "prompt_tokens"
        ),
        usage.get(
            "completion_tokens"
        ),
        usage.get(
            "total_tokens"
        ),
    )

    content = (
        data["choices"][0]
        ["message"]["content"]
        .strip()
    )

    llm_logger.info(
        "Raw memory evaluator response:\n%s",
        content,
    )

    if "</think>" in content:
        content = content.split(
            "</think>",
            1,
        )[1].strip()

    if content.startswith(
        "```"
    ):
        content = (
            content
            .strip("`")
            .strip()
        )

        if content.lower().startswith(
            "json"
        ):
            content = (
                content[4:]
                .strip()
            )

    content = (
        content.strip()
    )

    if (
        content.startswith("{")
        and content.count("{")
        == content.count("}") + 1
    ):
        content += "}"

    try:
        result = json.loads(
            content
        )

    except json.JSONDecodeError:
        memory_logger.warning(
            "Memory extraction returned "
            "invalid JSON content=%r",
            content,
        )

        return None

    if not result.get(
        "remember"
    ):
        memory_logger.info(
            "Memory evaluator rejected "
            "message remember=false"
        )

        return None

    memory_logger.info(
        "Memory evaluator accepted memory "
        "type=%s subject=%s "
        "predicate=%s importance=%s "
        "confidence=%s",
        result.get(
            "memory_type"
        ),
        result.get(
            "subject"
        ),
        result.get(
            "predicate"
        ),
        result.get(
            "importance"
        ),
        result.get(
            "confidence"
        ),
    )

    return result


# ---------------------------------------------------------------------
# Memory storage
# ---------------------------------------------------------------------

def save_extracted_memory(
    memory: dict,
) -> int | None:
    summary = (
        memory.get(
            "summary",
            "",
        )
        .strip()
    )

    if not summary:
        memory_logger.warning(
            "Skipping extracted memory "
            "with empty summary"
        )
        return None

    allowed_types = {
        "semantic",
        "episodic",
        "preference",
        "self",
        "relationship",
        "plan",
    }

    memory_type = (
        memory.get(
            "memory_type"
        )
    )

    if memory_type not in allowed_types:
        memory_logger.warning(
            "Skipping memory with "
            "invalid type=%s",
            memory_type,
        )
        return None

    try:
        importance = int(
            memory.get(
                "importance",
                5,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        importance = 5

    importance = max(
        1,
        min(
            10,
            importance,
        ),
    )

    try:
        confidence = float(
            memory.get(
                "confidence",
                1.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        confidence = 1.0

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    memory_logger.info(
        "Generating embedding for "
        "memory summary_chars=%d",
        len(summary),
    )

    embedding_started = (
        time.perf_counter()
    )

    embedding = embed_text(
        summary
    )

    embedding_elapsed = (
        time.perf_counter()
        - embedding_started
    )

    memory_logger.info(
        "Memory embedding generated "
        "elapsed=%.3fs dimensions=%d",
        embedding_elapsed,
        len(embedding),
    )

    memory_id = (
        memory_store.store_memory(
            memory_type=memory_type,
            subject=memory.get(
                "subject"
            ),
            predicate=memory.get(
                "predicate"
            ),
            object_text=memory.get(
                "object_text"
            ),
            summary=summary,
            importance=importance,
            confidence=confidence,
            embedding=embedding,
        )
    )

    memory_logger.info(
        "Memory created "
        "id=%s type=%s importance=%d "
        "confidence=%.2f summary=%r",
        memory_id,
        memory_type,
        importance,
        confidence,
        summary,
    )

    return memory_id


# ---------------------------------------------------------------------
# Background memory processing
# ---------------------------------------------------------------------

async def process_memory_background(
    user_message: str,
) -> None:
    """
    Evaluate and store long-term memory after the visible chat response
    has already been returned to the browser.

    Any failure here is logged but must never break the completed
    conversation response.
    """

    started = (
        time.perf_counter()
    )

    try:
        memory_logger.info(
            "Background memory processing started "
            "message_chars=%d",
            len(user_message),
        )

        extracted_memory = (
            await extract_memory(
                user_message
            )
        )

        memory_created = None

        if extracted_memory:
            memory_created = (
                save_extracted_memory(
                    extracted_memory
                )
            )

        elapsed = (
            time.perf_counter()
            - started
        )

        memory_logger.info(
            "Background memory processing completed "
            "elapsed=%.3fs memory_created=%s",
            elapsed,
            memory_created,
        )

    except Exception:
        elapsed = (
            time.perf_counter()
            - started
        )

        memory_logger.exception(
            "Background memory processing failed "
            "elapsed=%.3fs",
            elapsed,
        )


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------

@app.post(
    "/api/chat"
)
async def chat(
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    started = (
        time.perf_counter()
    )

    api_logger.info(
        "Chat request received "
        "user_id=%s "
        "history_messages=%d "
        "message_chars=%d",
        USER_ID,
        len(
            chat_request.history
        ),
        len(
            chat_request.message
        ),
    )

    try:
        # -------------------------------------------------------------
        # Persist Toby's message.
        # -------------------------------------------------------------

        user_message_id = (
            save_message(
                conversation_id=CONVERSATION_ID,
                role="user",
                content=chat_request.message,
            )
        )

        # -------------------------------------------------------------
        # Build Sable's system prompt.
        # -------------------------------------------------------------

        system_prompt = (
            load_system_prompt()
        )

        # -------------------------------------------------------------
        # Retrieve relevant long-term memories.
        # -------------------------------------------------------------

        memory_started = (
            time.perf_counter()
        )

        memory_logger.info(
            "Memory retrieval started "
            "query_chars=%d limit=%d",
            len(
                chat_request.message
            ),
            3,
        )

        memories = (
            retrieve_memories(
                chat_request.message,
                limit=3,
            )
        )

        memory_elapsed = (
            time.perf_counter()
            - memory_started
        )

        memory_logger.info(
            "Memory retrieval completed "
            "elapsed=%.3fs results=%d",
            memory_elapsed,
            len(memories),
        )

        memory_context = (
            build_memory_context(
                memories
            )
        )

        # -------------------------------------------------------------
        # Build conversation sent to llama-server.
        # -------------------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": memory_context,
                }
            )

        messages.extend(
            chat_request.history
        )

        messages.append(
            {
                "role": "user",
                "content": (
                    chat_request.message
                ),
            }
        )

        total_prompt_chars = sum(
            len(
                str(
                    message.get(
                        "content",
                        "",
                    )
                )
            )
            for message in messages
        )

        prompt_logger.info(
            "Chat prompt assembled "
            "messages=%d "
            "system_chars=%d "
            "memory_chars=%d "
            "history_messages=%d "
            "total_chars=%d",
            len(messages),
            len(system_prompt),
            len(memory_context),
            len(
                chat_request.history
            ),
            total_prompt_chars,
        )

        # -------------------------------------------------------------
        # Main generation.
        # -------------------------------------------------------------

        payload = {
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 4096,
            "stream": False,
        }

        llm_started = (
            time.perf_counter()
        )

        llm_logger.info(
            "Main generation started "
            "messages=%d "
            "prompt_chars=%d "
            "temperature=%.1f "
            "max_tokens=%d",
            len(messages),
            total_prompt_chars,
            payload["temperature"],
            payload["max_tokens"],
        )

        async with httpx.AsyncClient(
            timeout=300
        ) as client:
            response = (
                await client.post(
                    LLAMA_URL,
                    json=payload,
                )
            )

            response.raise_for_status()

            data = (
                response.json()
            )

        llm_elapsed = (
            time.perf_counter()
            - llm_started
        )

        usage = data.get(
            "usage",
            {},
        )

        raw_reply = (
            data["choices"][0]
            ["message"]["content"]
        )

        llm_logger.info(
            "Raw LLM response:\n%s",
            raw_reply,
        )

        reply = clean_llm_reply(
            raw_reply
        )

        api_logger.info(
            "Visible Sable reply:\n%s",
            reply,
        )

        llm_logger.info(
            "Main generation completed "
            "elapsed=%.3fs "
            "prompt_tokens=%s "
            "completion_tokens=%s "
            "total_tokens=%s "
            "raw_reply_chars=%d "
            "visible_reply_chars=%d",
            llm_elapsed,
            usage.get(
                "prompt_tokens"
            ),
            usage.get(
                "completion_tokens"
            ),
            usage.get(
                "total_tokens"
            ),
            len(raw_reply),
            len(reply),
        )

        # -------------------------------------------------------------
        # Persist only Sable's visible response.
        # -------------------------------------------------------------

        assistant_message_id = (
            save_message(
                conversation_id=CONVERSATION_ID,
                role="assistant",
                content=reply,
            )
        )

        # -------------------------------------------------------------
        # Queue durable-memory evaluation.
        #
        # This deliberately happens AFTER Sable's response has been
        # generated and saved. FastAPI runs the task after sending the
        # HTTP response, so the browser no longer waits for memory
        # extraction.
        # -------------------------------------------------------------

        background_tasks.add_task(
            process_memory_background,
            chat_request.message,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        api_logger.info(
            "Chat response ready "
            "user_id=%s "
            "elapsed=%.3fs "
            "memories_used=%d "
            "memory_evaluation=queued "
            "user_message_id=%s "
            "assistant_message_id=%s "
            "reply_chars=%d",
            USER_ID,
            elapsed,
            len(memories),
            user_message_id,
            assistant_message_id,
            len(reply),
        )

        return {
            "reply": reply,
            "user_id": USER_ID,
            "memories_used": memories,
            "memory_created": None,
            "memory_evaluation": "queued",
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
        }

    except Exception:
        elapsed = (
            time.perf_counter()
            - started
        )

        api_logger.exception(
            "Chat request failed "
            "user_id=%s elapsed=%.3fs",
            USER_ID,
            elapsed,
        )

        raise
