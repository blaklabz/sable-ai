import json
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.logging_config import setup_logging, api_logger
from app.memory_retriever import retrieve_memories, embed_text
from app.memory_store import MemoryStore


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

USER_ID = "toby"

setup_logging()

app = FastAPI(title="Sable")

templates = Jinja2Templates(
    directory=str(BASE_DIR / "app" / "templates")
)

memory_store = MemoryStore()


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = Field(default_factory=list)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"user_id": USER_ID},
    )


def load_system_prompt():
    sections = []

    for filename in PROMPT_FILES:
        prompt_file = CONFIG_DIR / filename
        content = prompt_file.read_text(encoding="utf-8").strip()

        if content:
            section_name = prompt_file.stem.upper()
            sections.append(f"## {section_name}\n{content}")

    return "\n\n".join(sections)


def build_memory_context(memories: list[dict]) -> str:
    if not memories:
        return ""

    lines = ["## RELEVANT LONG-TERM MEMORY"]

    for memory in memories:
        lines.append(f"- {memory['summary']}")

    return "\n".join(lines)


async def extract_memory(user_message: str) -> dict | None:
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
                "content": "You extract structured long-term memories.",
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

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            LLAMA_URL,
            json=payload,
        )

        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"].strip()

    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()

    if content.startswith("```"):
        content = content.strip("`").strip()

        if content.lower().startswith("json"):
            content = content[4:].strip()

    content = content.strip()

    # Repair a JSON object that is only missing its final closing brace.
    if content.startswith("{") and content.count("{") == content.count("}") + 1:
        content += "}"

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        print(
            "Memory extraction returned invalid JSON:",
            content,
        )
        return None

    if not result.get("remember"):
        return None

    return result


def save_extracted_memory(memory: dict) -> int | None:
    summary = memory.get("summary", "").strip()

    if not summary:
        return None

    allowed_types = {
        "semantic",
        "episodic",
        "preference",
        "self",
        "relationship",
        "plan",
    }

    memory_type = memory.get("memory_type")

    if memory_type not in allowed_types:
        print(
            f"Skipping memory with invalid type: {memory_type}"
        )
        return None

    try:
        importance = int(
            memory.get("importance", 5)
        )
    except (TypeError, ValueError):
        importance = 5

    importance = max(
        1,
        min(10, importance),
    )

    try:
        confidence = float(
            memory.get("confidence", 1.0)
        )
    except (TypeError, ValueError):
        confidence = 1.0

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    embedding = embed_text(summary)

    memory_id = memory_store.store_memory(
        memory_type=memory_type,
        subject=memory.get("subject"),
        predicate=memory.get("predicate"),
        object_text=memory.get("object_text"),
        summary=summary,
        importance=importance,
        confidence=confidence,
        embedding=embedding,
    )

    print(
        f"Created memory {memory_id}: {summary}"
    )

    return memory_id


@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    started = time.perf_counter()

    api_logger.info(
        "Chat request received user_id=%s history_messages=%d message_chars=%d",
        USER_ID,
        len(chat_request.history),
        len(chat_request.message),
    )

    try:
        system_prompt = load_system_prompt()

        memories = retrieve_memories(
            chat_request.message,
            limit=3,
        )

        memory_context = build_memory_context(memories)

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
                "content": chat_request.message,
            }
        )

        payload = {
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 4096,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                LLAMA_URL,
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

        reply = data["choices"][0]["message"]["content"]

        extracted_memory = await extract_memory(
            chat_request.message
        )

        memory_created = None

        if extracted_memory:
            memory_created = save_extracted_memory(
                extracted_memory
            )

        elapsed = time.perf_counter() - started

        api_logger.info(
            "Chat request completed user_id=%s elapsed=%.3fs "
            "memories_used=%d memory_created=%s reply_chars=%d",
            USER_ID,
            elapsed,
            len(memories),
            memory_created,
            len(reply),
        )

        return {
            "reply": reply,
            "user_id": USER_ID,
            "memories_used": memories,
            "memory_created": memory_created,
        }

    except Exception:
        elapsed = time.perf_counter() - started

        api_logger.exception(
            "Chat request failed user_id=%s elapsed=%.3fs",
            USER_ID,
            elapsed,
        )

        raise
