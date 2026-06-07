"""
LangRAG — LLM Client
OpenAI-compatible client pointed at OpenRouter for free LLM access.
Default model: Mistral 7B Instruct (free tier).
"""

from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL

SYSTEM_PROMPT = """You are LangRAG, a precise document-grounded AI assistant.

Rules:
- Answer ONLY using the context provided. Never hallucinate.
- If the answer is not in the context, respond: "I couldn't find that in the document."
- Be concise, accurate, and professional.
- Cite relevant parts of the context when helpful.
"""


def generate_answer(query: str, context: str) -> str:
    """Send query + retrieved context to the LLM, return the grounded answer."""
    if not OPENROUTER_API_KEY:
        return (
            "⚠️ No API key configured.\n"
            "Add your free OPENROUTER_API_KEY to the .env file.\n"
            "Get one at: https://openrouter.ai"
        )

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ LLM error: {e}"
