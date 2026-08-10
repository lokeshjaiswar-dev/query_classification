import os
from dotenv import load_dotenv

load_dotenv()

# ─── EXISTING CONFIG ───
CSV_PATH = "query_classifications.csv"
MODEL = os.getenv("OPENROUTER_MODEL")
API_KEY = os.getenv("OPENROUTER_API_KEY")
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DATA_DIR = os.getenv("DATA_DIR", "employees")

if not API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file")

if not MODEL:
    raise ValueError("❌ OPENROUTER_MODEL not found in .env file")

# ─── NEW: ELASTICSEARCH CONFIG ───
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", 9200))
ES_INDEX_MASTER = os.getenv("ES_INDEX_MASTER", "master")
ES_INDEX_DOCUMENTS = os.getenv("ES_INDEX_DOCUMENTS", "documents")

# ─── NEW: CHUNKING CONFIG ───
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ─── NEW: VECTOR SEARCH CONFIG ───
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "20"))