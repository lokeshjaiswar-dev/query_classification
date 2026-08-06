import os
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURATION ───
CSV_PATH = "query_classifications.csv"
MODEL = os.getenv("OPENROUTER_MODEL")
API_KEY = os.getenv("OPENROUTER_API_KEY")
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

if not API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file")

if not MODEL:
    raise ValueError("❌ OPENROUTER_MODEL not found in .env file")