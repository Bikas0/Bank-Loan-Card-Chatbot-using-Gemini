from pathlib import Path
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

LLM_MODEL = "models/gemini-flash-latest"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.30

# BASE_DIR = Path.cwd()
# FAISS_INDEX_PATH = BASE_DIR / "faiss.index"
# FAISS_META_PATH = BASE_DIR / "metadata.json"