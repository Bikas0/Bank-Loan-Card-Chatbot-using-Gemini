import os
import json
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

# Google GenAI SDK
import google.generativeai as genai

# Embedding client (same as indexing)
from embed_to_faiss import GeminiEmbeddingClient

# ===============================
# LOAD ENV
# ===============================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# ===============================
# CONFIG
# ===============================
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 1536          # MUST match FAISS index
TOP_K = 5
SIMILARITY_THRESHOLD = 0.30   # Tune: 0.25–0.40

BASE_DIR = Path.cwd()
FAISS_INDEX_PATH = BASE_DIR / "faiss.index"
FAISS_META_PATH = BASE_DIR / "metadata.json"

# ===============================
# VECTOR SEARCH (COSINE SIMILARITY)
# ===============================
class VectorSearch:
    def __init__(self, index_path: Path, metadata_path: Path):
        if not index_path.exists():
            raise FileNotFoundError("❌ FAISS index not found")
        if not metadata_path.exists():
            raise FileNotFoundError("❌ Metadata file not found")

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if self.index.d != EMBEDDING_DIM:
            raise ValueError(
                f"❌ FAISS dim ({self.index.d}) != EMBEDDING_DIM ({EMBEDDING_DIM})"
            )

        print(f"✅ FAISS cosine index loaded (dim={self.index.d})")

    def search(self, query_vector: np.ndarray):
        # FAISS requires (1, dim)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, TOP_K)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score >= SIMILARITY_THRESHOLD:
                results.append({
                    "content": self.metadata[idx]["content"],
                    "score": float(score),
                })

        return results

# ===============================
# GENERATIVE MODEL (FIXED PROMPT)
# ===============================
class GeminiFlash:
    def __init__(self):
        self.model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

    def answer(self, question: str, context: list[str]) -> str:
        context_text = "\n\n".join(context)

        # 🔥 CORRECT RAG PROMPT (THIS FIXES YOUR ISSUE)
        prompt = f"""
You are an AI assistant.

Use the information in the context below to answer the question.
If the context contains relevant information, explain or summarize it clearly.
If the context is not relevant to the question, say "I don't know".

Context:
{context_text}

Question:
{question}
"""

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 2048,
            }
        )

        if not response or not response.text:
            return "I don't know"

        return response.text.strip()

# ===============================
# QA SYSTEM
# ===============================
class QASystem:
    def __init__(self):
        self.embedder = GeminiEmbeddingClient(GEMINI_API_KEY)
        self.vector_search = VectorSearch(
            FAISS_INDEX_PATH,
            FAISS_META_PATH
        )
        self.generator = GeminiFlash()

    def ask(self, question: str) -> str:
        print("🔎 Embedding query...")

        query_vec = np.array(
            self.embedder.embed(question),
            dtype="float32"
        )

        if query_vec.size == 0:
            raise ValueError("❌ Empty embedding returned")

        print("📂 Semantic similarity search...")
        results = self.vector_search.search(query_vec)

        if not results:
            print("⚠️ No semantically similar content found")
            return "I don't know"

        print("\n🔎 Top semantic matches:")
        for r in results:
            print(f"  • cosine_score = {r['score']:.3f}")

        context = [r["content"] for r in results]

        print("🤖 Generating answer...")
        return self.generator.answer(question, context)

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    qa_system = QASystem()

    while True:
        question = input("\n📝 Enter your question (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        answer = qa_system.ask(question)

        print("\n" + "=" * 80)
        print("QUESTION:")
        print(question)
        print("\nANSWER:")
        print(answer)
        print("=" * 80)
