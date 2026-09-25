"""Turn cleaned restaurant records into text chunks + metadata for the vector store.

Each restaurant produces:
  * one "profile" chunk (facts: location, cuisines, cost, rating, services, popular dishes)
  * zero or more "reviews" chunks (customer review excerpts, split to ~REVIEW_CHUNK_CHARS)

Every chunk starts with a short header (name, location, cuisines) so it stays meaningful
on its own, and every chunk carries the full restaurant metadata so filters apply to all.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from restaurant_bot import config


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _clean_number(value) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def build_metadata(r: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata must be scalars or non-empty lists; missing numbers become -1."""
    rating = _clean_number(r.get("rating"))
    cost = _clean_number(r.get("cost_for_two"))
    meta = {
        "restaurant_id": r["restaurant_id"],
        "name": r["name"],
        "address": r["address"],
        "location": r["location"],
        "city": r.get("city") or "",
        "rest_type": r.get("rest_type") or "",
        "cuisines_text": ", ".join(r.get("cuisines") or []),
        "rating": rating if rating is not None else -1.0,
        "votes": int(r.get("votes") or 0),
        "cost_for_two": int(cost) if cost is not None else -1,
        "online_order": bool(r.get("online_order")),
        "book_table": bool(r.get("book_table")),
        "pure_veg": bool(r.get("pure_veg")),
        "categories_text": ", ".join(r.get("categories") or []),
        "url": r.get("url") or "",
    }
    # List metadata enables exact `$contains` filtering in Chroma. Empty lists are not allowed.
    if r.get("cuisines"):
        meta["cuisines"] = list(r["cuisines"])
    return meta


def _header(r: Dict[str, Any]) -> str:
    cuisines = ", ".join(r.get("cuisines") or []) or "unknown cuisine"
    return f"{r['name']} ({r.get('rest_type') or 'Restaurant'}) in {r['location']}, Bangalore. Cuisines: {cuisines}."


def profile_text(r: Dict[str, Any]) -> str:
    rating = _clean_number(r.get("rating"))
    cost = _clean_number(r.get("cost_for_two"))
    lines = [
        _header(r),
        f"Address: {r['address']}.",
        f"Rating: {rating}/5 from {int(r.get('votes') or 0)} votes." if rating is not None
        else "Rating: not yet rated (new restaurant).",
        f"Approximate cost for two people: Rs. {int(cost)}." if cost is not None
        else "Approximate cost for two: not listed.",
        f"Listed under: {', '.join(r.get('categories') or []) or 'n/a'}.",
        f"Online ordering: {'yes' if r.get('online_order') else 'no'}. "
        f"Table booking: {'yes' if r.get('book_table') else 'no'}.",
    ]
    if r.get("pure_veg"):
        lines.append("Described as a pure vegetarian restaurant.")
    if r.get("dishes_liked"):
        lines.append(f"Popular dishes: {', '.join(r['dishes_liked'])}.")
    return "\n".join(lines)


def split_reviews(reviews: List[str], max_chars: int) -> List[str]:
    """Greedily pack whole review excerpts into chunks of at most ~max_chars."""
    chunks, current = [], ""
    for review in reviews:
        piece = f'- "{review}"'
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def restaurant_to_chunks(r: Dict[str, Any]) -> List[Chunk]:
    meta = build_metadata(r)
    rid = r["restaurant_id"]
    chunks = [Chunk(id=f"{rid}-profile", text=profile_text(r), metadata={**meta, "chunk_type": "profile"})]
    for i, body in enumerate(split_reviews(r.get("reviews") or [], config.REVIEW_CHUNK_CHARS)):
        text = f"{_header(r)}\nCustomer reviews:\n{body}"
        chunks.append(Chunk(id=f"{rid}-reviews-{i}", text=text, metadata={**meta, "chunk_type": "reviews"}))
    return chunks


def build_chunks(records: Iterable[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for r in records:
        chunks.extend(restaurant_to_chunks(r))
    return chunks
