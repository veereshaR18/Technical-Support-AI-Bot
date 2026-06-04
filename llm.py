"""Part B2: DeepInfra/OpenAI-compatible LLM integration and prompting."""

from __future__ import annotations

import os

from openai import OpenAI

from config import DEEPINFRA_BASE_URL, DEEPINFRA_MODEL


HALLUCINATION_FALLBACK = (
    "I'm sorry, but the provided documentation does not contain that information."
)

SYSTEM_PROMPT = f"""You are a Senior Upwork API Consultant.

Answer developer questions using only the documentation excerpts provided in the
user message.

Rules:
1. Do not use outside knowledge.
2. If the excerpts do not contain the answer, reply exactly:
   "{HALLUCINATION_FALLBACK}"
3. Be precise. Include exact TTLs, grant names, endpoint names, parameters, or
   limits when they appear in the excerpts.
4. Keep the answer concise and practical.
"""


def get_llm_client() -> OpenAI:
    """Create a DeepInfra client without hardcoding secrets."""
    api_key = (os.getenv("DEEPINFRA_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise EnvironmentError(
            "DEEPINFRA_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    if api_key.startswith("http://") or api_key.startswith("https://"):
        raise EnvironmentError(
            "DEEPINFRA_API_KEY contains a URL. Paste the actual DeepInfra API token, "
            "not the endpoint URL. The endpoint is already configured in config.py."
        )

    return OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)


def build_user_message(query: str, chunks: list[str]) -> str:
    """Combine retrieved documentation snippets with the developer's question."""
    context = "\n\n---\n\n".join(
        f"[Source {index}]\n{chunk}" for index, chunk in enumerate(chunks, start=1)
    )
    return f"DOCUMENTATION EXCERPTS:\n{context}\n\nQUESTION:\n{query}"


def generate_answer(query: str, chunks: list[str]) -> str:
    """Call the hosted LLM and return its answer."""
    client = get_llm_client()
    response = client.chat.completions.create(
        model=DEEPINFRA_MODEL,
        temperature=0.1,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(query, chunks)},
        ],
    )
    return response.choices[0].message.content.strip()


