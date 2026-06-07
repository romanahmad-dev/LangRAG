"""
LangRAG Configuration
Centralised settings loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenRouter (Free LLM API) ──────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")

# ── Embeddings (local, free) ───────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Vector Store ───────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"

# ── RAG Settings ───────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 5

# ── App ────────────────────────────────────────────────────────────────────────
APP_TITLE       = "LangRAG — Document Intelligence"
APP_DESCRIPTION = "LangChain-powered RAG chatbot. Upload a PDF or TXT and ask questions grounded in your document."
