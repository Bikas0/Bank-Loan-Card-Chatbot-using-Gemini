import os
import json
import requests
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# ===============================
# LOAD ENV
# ===============================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file")

# ===============================
# CONFIG
# ===============================
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 1536  # ✅ REQUIRED

# Chunking settings to match Odoo AI module
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 5000

# Save exactly where script runs
BASE_DIR = Path(__file__).parent
FAISS_INDEX_PATH = BASE_DIR / "faiss.index"
FAISS_META_PATH = BASE_DIR / "metadata.json"

# ===============================
# GEMINI EMBEDDING CLIENT
# ===============================
class GeminiEmbeddingClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
                "dimensions": EMBEDDING_DIM,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

# ===============================
# TEXT CHUNKING
# ===============================
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
               min_chunk_size=MIN_CHUNK_SIZE, max_chunk_size=MAX_CHUNK_SIZE) -> list[str]:
    """
    Split text into chunks similar to Odoo AI module logic.
    """
    text = text.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n').strip()
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    def _add_chunk(chunk_text):
        if len(chunk_text) <= max_chunk_size:
            chunks.append(chunk_text)
        else:
            # Split oversized chunk by words
            words = chunk_text.split()
            temp = []
            temp_len = 0
            for word in words:
                wl = len(word) + 1
                if temp_len + wl > max_chunk_size:
                    if temp:
                        chunks.append(" ".join(temp))
                    temp = [word]
                    temp_len = len(word)
                else:
                    temp.append(word)
                    temp_len += wl
            if temp:
                chunks.append(" ".join(temp))

    for para in paragraphs:
        para_len = len(para)
        if current_chunk and (current_length + para_len + 1 <= chunk_size + overlap):
            current_chunk.append(para)
            current_length += para_len + 1
        else:
            if current_chunk:
                _add_chunk(" ".join(current_chunk))
            current_chunk = [para]
            current_length = para_len

    if current_chunk:
        last_chunk = " ".join(current_chunk)
        if chunks and len(last_chunk) < min_chunk_size:
            if len(chunks[-1]) + len(last_chunk) + 1 <= max_chunk_size:
                chunks[-1] += " " + last_chunk
            else:
                _add_chunk(last_chunk)
        else:
            _add_chunk(last_chunk)

    return chunks

# ===============================
# FAISS STORE
# ===============================
class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, vectors: list[list[float]], metas: list[dict]):
        vectors = np.array(vectors, dtype="float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metas)

    def save(self):
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

# ===============================
# MAIN PIPELINE
# ===============================
def embed_txt_file(txt_file: Path):
    if not txt_file.exists():
        raise FileNotFoundError(
            f"❌ File not found: {txt_file}\nMake sure the file exists. Current folder: {BASE_DIR}"
        )

    print(f"📖 Reading: {txt_file.resolve()}")
    text = txt_file.read_text(encoding="utf-8")

    chunks = chunk_text(text)
    print(f"✂️  Total chunks: {len(chunks)}")

    embedder = GeminiEmbeddingClient(GEMINI_API_KEY)
    store = FaissStore(EMBEDDING_DIM)

    vectors = []
    metas = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"🔢 Embedding chunk {i}/{len(chunks)}")
        vectors.append(embedder.embed(chunk))
        metas.append({
            "chunk_id": i,
            "content": chunk,
            "source": str(txt_file),
        })

    store.add(vectors, metas)
    store.save()

    print("\n✅ EMBEDDING COMPLETE")
    print(f"📦 FAISS index  : {FAISS_INDEX_PATH}")
    print(f"📄 Metadata    : {FAISS_META_PATH}")

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    txt_file = BASE_DIR / "Dataset" / "bank_data_snapshot.txt"
    embed_txt_file(txt_file)

