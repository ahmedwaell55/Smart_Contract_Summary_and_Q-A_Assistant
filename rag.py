import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np
import requests
from docx import Document
from pypdf import PdfReader


@dataclass
class RetrievedChunk:
    label: str
    text: str

# main class
class ContractRAG:
    def __init__(
        self,
        openrouter_api_key: str,
        openrouter_embed_model: str,
        groq_api_key: str,
        groq_model: str,
        groq_base_url: str,
        chunk_size: int = 1200,
        overlap: int = 200,
        top_k: int = 8,
    ) -> None:
        if not openrouter_api_key:
            raise ValueError("Missing OPENROUTER_API_KEY")
        if not openrouter_embed_model:
            raise ValueError("Missing OPENROUTER_EMBED_MODEL")
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY")
        if not groq_model:
            raise ValueError("Missing GROQ_LLM_MODEL")
        if not groq_base_url:
            raise ValueError("Missing GROQ_BASE_URL")

        self.openrouter_api_key = openrouter_api_key
        self.openrouter_embed_model = openrouter_embed_model
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.groq_base_url = groq_base_url.rstrip("/")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

        self.index = None
        self.dim = None
        self.chunks: List[str] = []

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _read_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return self._clean_text("\n".join(pages))

    def _read_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return self._clean_text("\n".join(p.text for p in doc.paragraphs))

    def _read_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._read_pdf(file_path)
        if ext == ".docx":
            return self._read_docx(file_path)
        raise ValueError(f"Unsupported file type: {ext}")

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        step = self.chunk_size - self.overlap
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step
        return chunks

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No text provided for embeddings")

        url = "https://openrouter.ai/api/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        vectors = []
        for text in texts:
            payload = {"model": self.openrouter_embed_model, "input": text}
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(f"Embedding request failed: {resp.text}") from exc
            data = resp.json()
            emb = data.get("data", [{}])[0].get("embedding")
            if not emb:
                raise RuntimeError("Embedding response missing vector")
            vectors.append(emb)

        arr = np.array(vectors, dtype="float32")
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise RuntimeError("Invalid embedding array shape")
        return arr

    def ingest_files(self, file_paths: List[str]) -> int:
        if not file_paths:
            raise ValueError("No files provided")

        all_chunks: List[str] = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in {".pdf", ".docx"}:
                continue
            text = self._read_file(file_path)
            if not text:
                continue
            all_chunks.extend(self._chunk_text(text))

        if not all_chunks:
            raise ValueError("No readable text found in uploaded files")

        vectors = self._embed_texts(all_chunks)
        faiss.normalize_L2(vectors)

        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.chunks = all_chunks

        return len(self.chunks)

    @staticmethod
    def _is_broad_question(question: str) -> bool:
        q = question.lower()
        return any(word in q for word in ["about", "summary", "purpose"])

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        if self.index is None or not self.chunks:
            raise ValueError("No indexed contract. Upload and process files first.")

        q_vec = self._embed_texts([question])
        faiss.normalize_L2(q_vec)

        k = min(self.top_k, len(self.chunks))
        _, indices = self.index.search(q_vec, k)

        hits = [self.chunks[i] for i in indices[0] if i >= 0]

        if self._is_broad_question(question) and self.chunks:
            first = self.chunks[0]
            hits = [first] + [h for h in hits if h != first]
            hits = hits[:k]

        return [RetrievedChunk(label=f"[{i + 1}]", text=chunk) for i, chunk in enumerate(hits)]

    def _build_messages(self, question: str, retrieved: List[RetrievedChunk]) -> List[dict]:
        context_lines = [f"{item.label} {item.text}" for item in retrieved]
        context = "\n\n".join(context_lines)

        system_prompt = (
            "You are a strict contract QA system. Use only the provided sources. "
            "Return exactly one short sentence and cite exactly one source tag like [1]. "
            "Never use outside knowledge. If the answer is not explicitly stated in the sources, "
            "return exactly: Not stated in the contract."
        )

        user_prompt = (
            f"Question: {question}\n\n"
            f"Sources:\n{context}\n\n"
            "Rules: one short sentence only, exactly one citation tag, no extra text."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _chat_completion(self, messages: List[dict]) -> str:
        url = f"{self.groq_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 60,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: {resp.text}") from exc

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Invalid LLM response format") from exc

    @staticmethod
    def _first_sentence(text: str) -> str:
        cleaned = " ".join((text or "").strip().split())
        if not cleaned:
            return ""
        match = re.search(r"[.!?]", cleaned)
        if not match:
            return cleaned
        return cleaned[: match.start() + 1]

    def _enforce_output(self, raw: str, retrieved: List[RetrievedChunk]) -> str:
        if "Not stated in the contract." in raw:
            return "Not stated in the contract."

        one_line = " ".join(raw.strip().splitlines()).strip()
        one_sentence = self._first_sentence(one_line)

        if not one_sentence:
            return "Not stated in the contract."

        tags = re.findall(r"\[(\d+)\]", one_sentence)
        if tags:
            chosen = int(tags[0])
        else:
            chosen = 1

        valid_ids = set(range(1, len(retrieved) + 1))
        if chosen not in valid_ids:
            chosen = 1

        text_no_tags = re.sub(r"\s*\[\d+\]", "", one_sentence).strip()
        text_no_tags = text_no_tags.rstrip(".") + "."

        return f"{text_no_tags} [{chosen}]"

    def answer_question(self, question: str) -> Tuple[str, List[RetrievedChunk]]:
        if not question or not question.strip():
            raise ValueError("Question is required")

        retrieved = self.retrieve(question)
        messages = self._build_messages(question, retrieved)
        raw = self._chat_completion(messages)
        final = self._enforce_output(raw, retrieved)
        return final, retrieved
