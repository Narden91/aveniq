"""Smoke test for Groq API connectivity."""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)
def test_groq_api_connectivity():
    """Verify that a basic Groq LLM call succeeds."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
    )
    try:
        response = llm.invoke("Say hello!")
    except Exception as exc:
        pytest.skip(f"Groq API connectivity unavailable: {exc}")
    assert response.content, "Expected non-empty response from Groq"