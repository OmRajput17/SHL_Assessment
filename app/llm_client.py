import os
from groq import Groq

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=api_key)
    return _client


def complete(system: str, user: str) -> str:
    """Call the Groq chat completions API and return the assistant message content."""
    resp = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=800,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content
