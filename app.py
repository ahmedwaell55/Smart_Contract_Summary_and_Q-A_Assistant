import os
from typing import List

import gradio as gr
from dotenv import load_dotenv

from rag import ContractRAG


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "qwen/qwen3-embedding-4b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

rag = ContractRAG(
    openrouter_api_key=OPENROUTER_API_KEY,
    openrouter_embed_model=OPENROUTER_EMBED_MODEL,
    groq_api_key=GROQ_API_KEY,
    groq_model=GROQ_LLM_MODEL,
    groq_base_url=GROQ_BASE_URL,
    chunk_size=1200,
    overlap=200,
    top_k=8,
)

indexed_signature = None


def _file_paths(files) -> List[str]:
    if not files:
        return []
    paths = []
    for f in files:
        path = getattr(f, "name", None)
        if path:
            paths.append(path)
    return paths


def ask_contract(files, question):
    global indexed_signature

    try:
        file_paths = _file_paths(files)
        if not file_paths:
            return "Upload at least one PDF or DOCX file."

        unsupported = [p for p in file_paths if not p.lower().endswith((".pdf", ".docx"))]
        if unsupported:
            return "Only PDF and DOCX files are supported."

        signature = tuple(sorted(file_paths))
        if signature != indexed_signature:
            rag.ingest_files(file_paths)
            indexed_signature = signature

        answer, _ = rag.answer_question(question)
        return answer
    except Exception as exc:
        return f"Error: {str(exc)}"


with gr.Blocks(title="Contract Q&A") as demo:
    gr.Markdown("# Smart Contract Summary & Q&A Assistant")

    files = gr.Files(label="Upload contract files (PDF/DOCX)", file_types=[".pdf", ".docx"])
    question = gr.Textbox(label="Question", placeholder="Ask about the contract")
    output = gr.Textbox(label="Answer", lines=2)
    ask_btn = gr.Button("Ask")

    ask_btn.click(fn=ask_contract, inputs=[files, question], outputs=output)


if __name__ == "__main__":
    demo.launch()
