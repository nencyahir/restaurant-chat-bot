"""Ingestion pipeline: raw CSV -> cleaned records -> chunks -> embeddings -> ChromaDB."""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from restaurant_bot import config, data_loader, documents, retriever, vector_store

logger = logging.getLogger(__name__)


def build_vocab(df: pd.DataFrame) -> Dict[str, List[str]]:
    cuisines = sorted({c for items in df["cuisines"] for c in items})
    locations = sorted({loc for loc in df["location"] if isinstance(loc, str) and loc})
    return {"cuisines": cuisines, "locations": locations}


def run(raw_path: Path = config.RAW_DATA_PATH, client=None, limit: int = None) -> Dict[str, int]:
    logger.info("Loading raw dataset from %s", raw_path)
    raw = data_loader.load_raw(raw_path)
    logger.info("Raw rows: %d", len(raw))

    cleaned = data_loader.clean(raw)
    if limit:
        cleaned = cleaned.head(limit)
    data_loader.save_processed(cleaned)
    retriever.save_vocab(build_vocab(cleaned))

    chunks = documents.build_chunks(cleaned.to_dict("records"))
    logger.info("Created %d chunks for %d restaurants", len(chunks), len(cleaned))

    collection = vector_store.build_collection(chunks, client=client)
    stats = {"raw_rows": len(raw), "restaurants": len(cleaned), "chunks": len(chunks),
             "stored": collection.count()}
    logger.info("Ingestion complete: %s", stats)
    return stats
