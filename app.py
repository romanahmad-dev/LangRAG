"""
LangRAG — Document Intelligence
Gradio-powered RAG chatbot built with LangChain + ChromaDB + OpenRouter.

Run:
    python app.py
"""

import os
import sys
import tempfile

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.config import APP_TITLE, APP_DESCRIPTION, CHROMA_PERSIST_DIR
from src.rag_pipeline import ingest, retrieve_context
from src.llm_client import generate_answer

# ── State ──────────────────────────────────────────────────────────────────────
vectordb     = None
current_doc  = None
ingest_stats = {}

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "Data", "new-Policies.txt")

# ── CSS ────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background: #0b0f1a !important;
    color: #e2e8f0 !important;
}

/* Header */
.langrag-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}
.langrag-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, #6366f180 0%, transparent 70%);
    pointer-events: none;
}
.langrag-title {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: #a5b4fc !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.5px;
}
.langrag-sub {
    font-size: 0.92rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.5;
}
.badge-row {
    margin-top: 14px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid;
}
.badge-indigo { color: #a5b4fc; border-color: #4338ca; background: #1e1b4b; }
.badge-emerald { color: #6ee7b7; border-color: #059669; background: #064e3b; }
.badge-amber { color: #fcd34d; border-color: #b45309; background: #451a03; }
.badge-rose { color: #fda4af; border-color: #be123c; background: #4c0519; }

/* Panels */
.panel {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 20px;
}
.panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 14px;
    font-weight: 600;
}

/* Stats box */
.stats-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #7dd3fc;
    line-height: 1.8;
    white-space: pre;
}

/* Pipeline steps */
.pipe-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    background: #1e293b;
    border-left: 3px solid #334155;
    transition: border-color 0.3s;
}
.pipe-step.active { border-left-color: #6366f1; color: #a5b4fc; }
.pipe-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #6366f1;
    min-width: 18px;
}

/* Chat */
.chatbot-wrap .message.user { background: #1e1b4b !important; border-radius: 10px !important; }
.chatbot-wrap .message.bot  { background: #0f2723 !important; border-radius: 10px !important; }

/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    color: white !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
}
.btn-primary:hover { opacity: 0.9 !important; }
.btn-secondary {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
}

/* Context accordion */
.context-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #fcd34d;
    background: #1c1400;
    border: 1px solid #78350f;
    border-radius: 8px;
    padding: 14px;
    white-space: pre-wrap;
    line-height: 1.6;
    max-height: 200px;
    overflow-y: auto;
}

/* Input */
input[type=password], input[type=text], textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Footer */
.footer-bar {
    text-align: center;
    padding: 16px;
    font-size: 0.78rem;
    color: #475569;
    font-family: 'IBM Plex Mono', monospace;
    border-top: 1px solid #1e293b;
    margin-top: 12px;
}
.footer-bar a { color: #6366f1; text-decoration: none; }
"""

# ── Pipeline HTML ──────────────────────────────────────────────────────────────
PIPELINE_HTML = """
<div class="panel">
  <div class="panel-title">⛓ RAG Pipeline</div>
  <div class="pipe-step"><span class="pipe-num">01</span> 📥 Load Document (PDF / TXT)</div>
  <div class="pipe-step"><span class="pipe-num">02</span> ✂️ Chunk with RecursiveTextSplitter</div>
  <div class="pipe-step"><span class="pipe-num">03</span> 🧠 Embed via HuggingFace MiniLM</div>
  <div class="pipe-step"><span class="pipe-num">04</span> 🗄️ Store in ChromaDB</div>
  <div class="pipe-step"><span class="pipe-num">05</span> 🔍 Retrieve Top-K Chunks</div>
  <div class="pipe-step"><span class="pipe-num">06</span> 🤖 Generate Answer via LLM</div>
</div>
"""

HEADER_HTML = """
<div class="langrag-header">
  <div class="langrag-title">⛓ LangRAG</div>
  <p class="langrag-sub">Document Intelligence — Retrieval-Augmented Generation with LangChain · ChromaDB · OpenRouter</p>
  <div class="badge-row">
    <span class="badge badge-indigo">LangChain</span>
    <span class="badge badge-emerald">ChromaDB</span>
    <span class="badge badge-amber">HuggingFace</span>
    <span class="badge badge-rose">OpenRouter · Free LLM</span>
    <span class="badge badge-indigo">IBM Certified</span>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="footer-bar">
  Built by <a href="https://roman-ai.replit.app" target="_blank">Roman Ahmad</a> ·
  AI Integration Engineer ·
  <a href="https://github.com/romanahmad-dev/botchain" target="_blank">GitHub</a> ·
  <a href="https://www.coursera.org/account/accomplishments/verify/XJD5NZZIS29O" target="_blank">IBM Certificate</a>
</div>
"""

# ── Core Functions ─────────────────────────────────────────────────────────────

def ingest_file(file_obj, api_key: str):
    """Ingest an uploaded file into ChromaDB."""
    global vectordb, current_doc, ingest_stats

    if api_key.strip():
        os.environ["OPENROUTER_API_KEY"] = api_key.strip()

    if file_obj is None:
        return "⚠️ No file uploaded.", ""

    file_path = file_obj.name
    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in (".pdf", ".txt"):
        return "❌ Only PDF and TXT files supported.", ""

    try:
        persist_dir = CHROMA_PERSIST_DIR + f"_{os.path.basename(file_path)[:12]}"
        vectordb, stats = ingest(file_path, persist_dir)
        current_doc = os.path.basename(file_path)
        ingest_stats = stats

        stats_text = (
            f"✅ Ingestion complete\n"
            f"─────────────────────\n"
            f"📄 File    : {current_doc}\n"
            f"📃 Pages   : {stats['pages']}\n"
            f"✂️  Chunks  : {stats['chunks']}\n"
            f"🧠 Embedder: all-MiniLM-L6-v2\n"
            f"🗄️  Store   : ChromaDB\n"
            f"─────────────────────\n"
            f"Ready to answer questions!"
        )
        return stats_text, current_doc

    except Exception as e:
        return f"❌ Ingestion error: {e}", ""


def load_sample(api_key: str):
    """Load the bundled Company Policies sample document."""
    global vectordb, current_doc, ingest_stats

    if api_key.strip():
        os.environ["OPENROUTER_API_KEY"] = api_key.strip()

    if not os.path.exists(SAMPLE_PATH):
        return "❌ Sample file not found at Data/new-Policies.txt", ""

    try:
        vectordb, stats = ingest(SAMPLE_PATH, CHROMA_PERSIST_DIR + "_sample")
        current_doc = "new-Policies.txt"
        ingest_stats = stats

        stats_text = (
            f"✅ Sample loaded\n"
            f"─────────────────────\n"
            f"📄 File    : new-Policies.txt\n"
            f"📃 Pages   : {stats['pages']}\n"
            f"✂️  Chunks  : {stats['chunks']}\n"
            f"🧠 Embedder: all-MiniLM-L6-v2\n"
            f"🗄️  Store   : ChromaDB\n"
            f"─────────────────────\n"
            f"Ask about: smoking, email, mobile, recruitment policies..."
        )
        return stats_text, current_doc

    except Exception as e:
        return f"❌ Error: {e}", ""


def chat(message: str, history: list):
    """Handle a chat turn: retrieve context + generate grounded answer."""
    global vectordb

    if not message.strip():
        return history, "", ""

    if vectordb is None:
        history.append((message, "⚠️ Please upload a document or load the sample first."))
        return history, "", ""

    context, docs = retrieve_context(vectordb, message)
    answer = generate_answer(message, context)

    history.append((message, answer))

    ctx_preview = f"Retrieved {len(docs)} chunk(s):\n\n" + context[:1200] + ("..." if len(context) > 1200 else "")
    return history, "", ctx_preview


def clear_chat():
    return [], "", ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(css=CUSTOM_CSS, title="LangRAG — Document Intelligence") as demo:

    # Header
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):

        # ── Left sidebar ───────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            gr.HTML('<div class="panel-title" style="margin-top:8px">🔑 API Key</div>')
            api_key = gr.Textbox(
                placeholder="sk-or-... (get free key at openrouter.ai)",
                type="password",
                show_label=False,
                container=False,
            )

            gr.HTML('<div class="panel-title" style="margin-top:16px">📄 Document</div>')
            file_upload = gr.File(
                label="Upload PDF or TXT",
                file_types=[".pdf", ".txt"],
            )

            with gr.Row():
                ingest_btn = gr.Button("⚡ Ingest Document", elem_classes=["btn-primary"])
                sample_btn = gr.Button("📋 Load Sample", elem_classes=["btn-secondary"])

            doc_name = gr.Textbox(
                label="Active Document",
                interactive=False,
                placeholder="None loaded",
                container=True,
            )

            stats_box = gr.Textbox(
                label="Pipeline Stats",
                interactive=False,
                lines=8,
                placeholder="Stats will appear after ingestion...",
                elem_classes=["stats-box"],
            )

            gr.HTML(PIPELINE_HTML)

        # ── Right chat area ────────────────────────────────────────────────────
        with gr.Column(scale=2):

            chatbot = gr.Chatbot(
                label="LangRAG Chat",
                height=440,
                show_copy_button=True,
                avatar_images=(None, "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png"),
                elem_classes=["chatbot-wrap"],
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question about your document...",
                    show_label=False,
                    container=False,
                    scale=5,
                )
                send_btn = gr.Button("Send ➤", elem_classes=["btn-primary"], scale=1)
                clear_btn = gr.Button("🗑", elem_classes=["btn-secondary"], scale=0)

            with gr.Accordion("🔍 Retrieved Context", open=False):
                context_box = gr.Textbox(
                    interactive=False,
                    show_label=False,
                    lines=6,
                    placeholder="Context retrieved from ChromaDB will appear here...",
                    elem_classes=["context-text"],
                )

            gr.Examples(
                examples=[
                    ["What is the smoking policy?"],
                    ["What are the rules for email use?"],
                    ["Explain the recruitment process."],
                    ["What happens if I lose my mobile phone?"],
                    ["What is the code of conduct?"],
                ],
                inputs=msg_box,
                label="💡 Example Questions (works with sample data)",
            )

    gr.HTML(FOOTER_HTML)

    # ── Event Wiring ───────────────────────────────────────────────────────────
    ingest_btn.click(
        ingest_file,
        inputs=[file_upload, api_key],
        outputs=[stats_box, doc_name],
    )

    sample_btn.click(
        load_sample,
        inputs=[api_key],
        outputs=[stats_box, doc_name],
    )

    send_btn.click(
        chat,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box, context_box],
    )

    msg_box.submit(
        chat,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box, context_box],
    )

    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg_box, context_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
