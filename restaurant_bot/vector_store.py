"""Embeddings (sentence-transformers) + persistent ChromaDB storage."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb

from restaurant_bot import config
from restaurant_bot.documents import Chunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when the vector store is missing or empty."""


@lru_cache(maxsize=2)
def get_embedder(model_name: str = config.EMBEDDING_MODEL):
    # Imported lazily: loading torch is slow and not needed for e.g. query parsing.
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s", model_name)
    return SentenceTransformer(model_name, device="cpu")


def embed(texts: Sequence[str], model_name: str = config.EMBEDDING_MODEL, batch_size: int = 64) -> List[List[float]]:
    model = get_embedder(model_name)
    vectors = model.encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 1000
    )
    return vectors.tolist()


def get_client(persist_dir: Path = config.CHROMA_DIR):
    return chromadb.PersistentClient(path=str(persist_dir))


def build_collection(chunks: List[Chunk], client=None, name: str = config.COLLECTION_NAME,
                     batch_size: int = 1000):
    """(Re)create the collection from scratch and store all chunks with their embeddings."""
    client = client or get_client()
    try:
        client.delete_collection(name)
    except Exception:  # collection did not exist yet
        pass
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=embed([c.text for c in batch]),
        )
        logger.info("Stored %d / %d chunks", min(start + batch_size, len(chunks)), len(chunks))
    return collection


def get_collection(client=None, name: str = config.COLLECTION_NAME):
    client = client or get_client()
    try:
        collection = client.get_collection(name)
    except Exception as exc:
        raise VectorStoreError(
            f"Vector store collection '{name}' not found. Run `python scripts/ingest.py` first."
        ) from exc
    if collection.count() == 0:
        raise VectorStoreError(f"Vector store collection '{name}' is empty. Re-run ingestion.")
    return collection


def query(collection, text: str, n_results: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Semantic search; returns a flat list of {id, text, metadata, distance}."""
    result = collection.query(
        query_embeddings=embed([text]),
        n_results=n_results,
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": i, "text": d, "metadata": m, "distance": dist}
        for i, d, m, dist in zip(result["ids"][0], result["documents"][0],
                                 result["metadatas"][0], result["distances"][0])
    ]
