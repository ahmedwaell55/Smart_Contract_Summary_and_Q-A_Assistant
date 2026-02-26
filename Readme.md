# Smart Contract Summary & Q&A Assistant

A Gradio-based Retrieval-Augmented Generation (RAG) app that answers questions about uploaded contract files (`.pdf` and `.docx`) using:
- OpenRouter embeddings for retrieval
- FAISS vector search for chunk matching
- Groq-hosted chat completion for final answer generation

## Features
- Upload one or more contract files (`PDF`/`DOCX`)
- Automatic text extraction and chunking
- Semantic retrieval over contract chunks
- One-sentence, citation-tagged answers (for example: `The governing law is New York law. [2]`)
- Strict fallback when information is missing: `Not stated in the contract.`

## Project Structure
- `app.py`: Gradio UI and request flow
- `rag.py`: Contract RAG pipeline (ingestion, embedding, retrieval, prompting, output enforcement)
- `requirements.txt`: Python dependencies

## Requirements
- Python `3.10+`
- API keys:
  - OpenRouter (embeddings)
  - Groq (LLM completion)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_EMBED_MODEL=qwen/qwen3-embedding-4b
GROQ_API_KEY=your_groq_api_key
GROQ_LLM_MODEL=llama-3.1-8b-instant
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

Notes:
- `OPENROUTER_API_KEY` and `GROQ_API_KEY` are required.
- Model/base URL values above are the current defaults used by the code.

## Run
```bash
python app.py
```

Then open the Gradio URL shown in the terminal.

## How It Works
1. Uploaded files are read (`pypdf` for PDF, `python-docx` for DOCX).
2. Text is normalized and split into overlapping chunks.
3. Each chunk is embedded via OpenRouter and indexed in FAISS.
4. A user question is embedded and top-k similar chunks are retrieved.
5. Retrieved chunks are sent to Groq with strict response rules.
6. Output is normalized to one short sentence with a single citation tag.

## Limitations
- Supports only `.pdf` and `.docx` uploads.
- Requires readable/extractable text (scanned image-only PDFs may fail).
- Re-indexing happens when uploaded file set changes.

## Troubleshooting
- `Missing OPENROUTER_API_KEY` or `Missing GROQ_API_KEY`:
  - Ensure `.env` exists and variables are set.
- `No readable text found in uploaded files`:
  - Verify documents contain extractable text.
- API request failures:
  - Check key validity, model names, and network connectivity.
