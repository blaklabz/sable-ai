from pathlib import Path
from app.memory_retriever import retrieve_memories


import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

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

app = FastAPI(title="Sable")

templates = Jinja2Templates(
    directory=str(BASE_DIR / "app" / "templates")
)


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


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


@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    system_prompt = load_system_prompt()

    memories = retrieve_memories(
        chat_request.message,
        limit=5,
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

    messages.extend(chat_request.history)

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
        response = await client.post(LLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    reply = data["choices"][0]["message"]["content"]

    return {
        "reply": reply,
        "user_id": USER_ID,
        "memories_used": memories,
    }


def build_memory_context(memories: list[dict]) -> str:
    if not memories:
        return ""

    lines = ["## RELEVANT LONG-TERM MEMORY"]

    for memory in memories:
        lines.append(f"- {memory['summary']}")

    return "\n".join(lines)
