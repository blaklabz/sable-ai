from pathlib import Path

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


@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    system_prompt = SYSTEM_PROMPT_FILE.read_text().strip()

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

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
    }
