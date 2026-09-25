"""Central configuration. Values can be overridden with environment variables (see .env.example)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_URL = (
    "https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation"
    "/resolve/main/zomato.csv"
)
# Full dataset (~575 MB, downloaded by scripts/download_data.py, not committed).
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", PROJECT_ROOT / "data" / "raw" / "zomato.csv"))
# Small committed sample: every listing row of the 500 most-voted restaurants (real data).
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "zomato_sample.csv"
PROCESSED_DATA_PATH = Path(
    os.getenv("PROCESSED_DATA_PATH", PROJECT_ROOT / "data" / "processed" / "restaurants.csv")
)
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "restaurants")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

# Chunking: review text is split into chunks of roughly this many characters.
REVIEW_CHUNK_CHARS = 700
MAX_REVIEWS_PER_RESTAURANT = 6
MAX_REVIEW_CHARS = 350

# Retrieval
DEFAULT_TOP_K = 5
CANDIDATE_MULTIPLIER = 6  # fetch more chunks than needed, then de-duplicate by restaurant
