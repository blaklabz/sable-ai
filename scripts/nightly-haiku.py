import asyncio
import time

import httpx

from app.logging_config import setup_logging, haiku_logger


LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"


async def generate_haiku() -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Sable. Write one original haiku. "
                    "The haiku may reflect on curiosity, technology, "
                    "nature, memory, bicycles, learning, or whatever "
                    "quiet thought feels appropriate. "
                    "Return only the three-line haiku. "
                    "Do not explain it. Do not show your reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write tonight's haiku.\n"
                    "/no_think"
                ),
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

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            LLAMA_URL,
            json=payload,
        )

        response.raise_for_status()
        data = response.json()

    elapsed = time.perf_counter() - started

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

    usage = data.get("usage", {})

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


async def main():
    setup_logging()

    try:
        haiku = await generate_haiku()

        print()
        print("Sable's nightly haiku:")
        print()
        print(haiku)
        print()

    except Exception:
        haiku_logger.exception(
            "Nightly haiku generation failed"
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())
