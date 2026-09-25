"""Semantic retrieval with metadata filtering, grouped into one result per restaurant."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from chromadb.errors import NotFoundError

from restaurant_bot import config, vector_store
from restaurant_bot.query_parser import Filters, merge_filters, parse_query, to_where

VOCAB_PATH = config.PROCESSED_DATA_PATH.parent / "vocab.json"


@dataclass
class RetrievedRestaurant:
    restaurant_id: str
    metadata: Dict[str, Any]
    profile: str
    evidence: List[str] = field(default_factory=list)  # matching review chunks
    similarity: float = 0.0


@dataclass
class RetrievalResult:
    query: str
    filters: Filters
    restaurants: List[RetrievedRestaurant]


def save_vocab(vocab: Dict[str, List[str]], path: Path = VOCAB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, indent=1))


def load_vocab(path: Path = VOCAB_PATH) -> Dict[str, List[str]]:
    if not path.exists():
        return {"cuisines": [], "locations": []}
    return json.loads(path.read_text())


class Retriever:
    def __init__(self, collection=None, vocab: Optional[Dict[str, List[str]]] = None):
        self._owns_collection = collection is None
        self.collection = collection if collection is not None else vector_store.get_collection()
        self.vocab = vocab if vocab is not None else load_vocab()

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K,
                 overrides: Optional[Filters] = None) -> RetrievalResult:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        try:
            return self._retrieve(query, top_k, overrides)
        except NotFoundError:
            # Re-running ingestion recreates the collection under a new id; reconnect once.
            if not self._owns_collection:
                raise
            self.collection = vector_store.get_collection()
            self.vocab = load_vocab()
            return self._retrieve(query, top_k, overrides)

    def _retrieve(self, query: str, top_k: int, overrides: Optional[Filters]) -> RetrievalResult:
        parsed = parse_query(query, self.vocab.get("cuisines", []), self.vocab.get("locations", []))
        filters = merge_filters(parsed, overrides)
        where = to_where(filters)

        # Over-fetch chunks: several chunks can belong to the same restaurant.
        # With sort_by_rating we fetch a wider pool so re-ranking by rating has room to work.
        multiplier = config.CANDIDATE_MULTIPLIER * (2 if filters.sort_by_rating else 1)
        hits = vector_store.query(self.collection, query, n_results=top_k * multiplier, where=where)

        grouped: Dict[str, RetrievedRestaurant] = {}
        for hit in hits:  # hits are ordered by similarity, best first
            meta = hit["metadata"]
            rid = meta["restaurant_id"]
            item = grouped.get(rid)
            if item is None:
                item = RetrievedRestaurant(restaurant_id=rid, metadata=meta, profile="",
                                           similarity=1 - hit["distance"])
                grouped[rid] = item
            if meta.get("chunk_type") == "profile":
                item.profile = hit["text"]
            else:
                item.evidence.append(hit["text"])

        restaurants = list(grouped.values())
        if filters.sort_by_rating:
            restaurants.sort(key=lambda r: (r.metadata.get("rating", -1), r.metadata.get("votes", 0)),
                             reverse=True)
        restaurants = restaurants[:top_k]
        self._fill_profiles(restaurants)
        return RetrievalResult(query=query, filters=filters, restaurants=restaurants)

    def _fill_profiles(self, restaurants: List[RetrievedRestaurant]) -> None:
        """Make sure each result carries its factual profile chunk, even if only a review matched."""
        missing = [f"{r.restaurant_id}-profile" for r in restaurants if not r.profile]
        if not missing:
            return
        got = self.collection.get(ids=missing, include=["documents"])
        by_id = dict(zip(got["ids"], got["documents"]))
        for r in restaurants:
            if not r.profile:
                r.profile = by_id.get(f"{r.restaurant_id}-profile", "")
