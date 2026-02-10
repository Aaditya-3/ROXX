"""
Groq LLM Client

Single source of truth for LLM calls.
Uses Groq only. No system roles. No streaming. No async. No retries.

Environment: GROQ_API_KEY must be set by the time this module is imported.
The backend entry file (main.py) loads .env at startup before importing this module.
"""

import os
from groq import Groq

# Model: env GROQ_MODEL or default llama-3.1-8b-instant
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"


def generate_response(prompt: str) -> str:
    """
    Send a single combined prompt to Groq and return plain text response.

    Args:
        prompt: Full prompt string (caller must include memory context if needed).

    Returns:
        Plain text response from the model.

    Raises:
        RuntimeError: If GROQ_API_KEY is missing or API call fails.
    """
    print("USING GROQ MODEL:", GROQ_MODEL)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError("GROQ_API_KEY not found in environment variables")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.55,
    )

    if not response or not response.choices:
        raise RuntimeError("Empty response from Groq")

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Empty response from Groq")

    return content.strip()
