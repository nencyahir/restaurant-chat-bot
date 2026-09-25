"""Rule-based extraction of structured filters (cuisine, location, budget, rating, ...) from a
natural-language question, and conversion of those filters into a ChromaDB `where` clause.

Cuisine and location names are matched against the vocabulary found in the dataset itself,
so a filter can never refer to a value that does not exist in the data.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

# Used when the user asks for "cheap"/"budget" food without giving an amount (Rs. for two).
DEFAULT_BUDGET = 500
HIGH_RATING = 4.0

_NUM = r"(?:rs\.?|inr|₹)?\s*(\d[\d,]*)"
MAX_COST_PATTERNS = [
    rf"\b(?:under|below|less than|within|upto|up to|max(?:imum)?|cheaper than|at most)\s*{_NUM}",
    rf"\bbudget(?: of| is)?\s*{_NUM}",
    rf"{_NUM}\s*(?:or less|and below|max)\b",
]
CHEAP_PATTERN = re.compile(r"\b(cheap|budget|affordable|inexpensive|pocket[- ]friendly|low[- ]cost)\b", re.I)
MIN_RATING_PATTERN = re.compile(
    r"\b(?:rat(?:ed|ing)|stars?)\s*(?:of\s*)?(?:above|over|at least|more than|>=?|\+)?\s*(\d(?:\.\d)?)\s*\+?"
    r"|\b(\d(?:\.\d)?)\s*\+?\s*(?:stars?|rating|rated)\b",
    re.I,
)
HIGH_RATING_PATTERN = re.compile(
    r"\b(highly|high|top|best|well|good)[- ]?(rated|rating|reviewed)\b|\bbest\b|\btop\b", re.I
)
VEG_PATTERN = re.compile(r"\b(pure[\s-]*)?veg(etarian|gie)?\b", re.I)
NON_VEG_PATTERN = re.compile(r"\bnon[\s-]*veg", re.I)
DELIVERY_PATTERN = re.compile(r"\b(deliver(y|s)?|order online|online order(ing)?|home delivery)\b", re.I)
BOOKING_PATTERN = re.compile(r"\b(book(ing)? a table|table booking|reserv(e|ation)|book table)\b", re.I)


@dataclass
class Filters:
    cuisines: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    max_cost: Optional[int] = None
    min_rating: Optional[float] = None
    pure_veg: bool = False
    online_order: bool = False
    book_table: bool = False
    sort_by_rating: bool = False

    def is_empty(self) -> bool:
        return not any([self.cuisines, self.locations, self.max_cost, self.min_rating,
                        self.pure_veg, self.online_order, self.book_table])

    def describe(self) -> Dict[str, Any]:
        """Only the filters that are actually applied, for display."""
        return {k: v for k, v in asdict(self).items() if v and k != "sort_by_rating"}


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])", text) is not None


def _base_location(location: str) -> str:
    """'Koramangala 5th Block' -> 'Koramangala'; 'JP Nagar' -> 'JP Nagar'."""
    return re.sub(r"\s+\d.*$", "", location).strip()


def match_cuisines(text: str, vocabulary: Iterable[str]) -> List[str]:
    q = text.lower()
    found: List[str] = []
    for cuisine in sorted(vocabulary, key=len, reverse=True):
        mentioned = _contains_phrase(q, cuisine) or _contains_phrase(q, cuisine + "s")  # 'cafes'
        if mentioned and not any(cuisine.lower() in f.lower() for f in found):
            found.append(cuisine)
    return found


def match_locations(text: str, vocabulary: Iterable[str]) -> List[str]:
    """Exact location names win; otherwise a base area name matches all its sub-locations."""
    q = text.lower()
    vocabulary = list(vocabulary)
    exact = [loc for loc in sorted(vocabulary, key=len, reverse=True) if _contains_phrase(q, loc)]
    if exact:
        longest = exact[0]
        # Keep additional exact matches only if they are not substrings of an earlier match.
        result = [longest] + [l for l in exact[1:] if l.lower() not in longest.lower()]
        return sorted(set(result))
    bases = {_base_location(loc) for loc in vocabulary}
    matched_bases = [b for b in bases if len(b) >= 3 and _contains_phrase(q, b)]
    return sorted(loc for loc in vocabulary if _base_location(loc) in matched_bases)


def _to_int(value: str) -> Optional[int]:
    digits = value.replace(",", "")
    return int(digits) if digits.isdigit() else None


def parse_query(text: str, cuisines_vocab: Iterable[str] = (), locations_vocab: Iterable[str] = ()) -> Filters:
    filters = Filters()
    q = text.lower()

    filters.cuisines = match_cuisines(q, cuisines_vocab)
    filters.locations = match_locations(q, locations_vocab)

    for pattern in MAX_COST_PATTERNS:
        match = re.search(pattern, q)
        if match:
            amount = _to_int(match.group(1))
            if amount and amount >= 50:  # ignore things like "under 5 minutes"-style noise
                filters.max_cost = amount
                break
    if filters.max_cost is None and CHEAP_PATTERN.search(q):
        filters.max_cost = DEFAULT_BUDGET

    rating_match = MIN_RATING_PATTERN.search(q)
    if rating_match:
        value = float(rating_match.group(1) or rating_match.group(2))
        if 0 < value <= 5:
            filters.min_rating = value
    if HIGH_RATING_PATTERN.search(q):
        filters.sort_by_rating = True
        if filters.min_rating is None:
            filters.min_rating = HIGH_RATING

    filters.pure_veg = bool(VEG_PATTERN.search(q)) and not NON_VEG_PATTERN.search(q)
    filters.online_order = bool(DELIVERY_PATTERN.search(q))
    filters.book_table = bool(BOOKING_PATTERN.search(q))
    return filters


def merge_filters(parsed: Filters, overrides: Optional[Filters]) -> Filters:
    """Explicit UI selections take precedence over values parsed from the text."""
    if overrides is None:
        return parsed
    return Filters(
        cuisines=overrides.cuisines or parsed.cuisines,
        locations=overrides.locations or parsed.locations,
        max_cost=overrides.max_cost if overrides.max_cost is not None else parsed.max_cost,
        min_rating=overrides.min_rating if overrides.min_rating is not None else parsed.min_rating,
        pure_veg=overrides.pure_veg or parsed.pure_veg,
        online_order=overrides.online_order or parsed.online_order,
        book_table=overrides.book_table or parsed.book_table,
        sort_by_rating=overrides.sort_by_rating or parsed.sort_by_rating,
    )


def to_where(filters: Filters) -> Optional[Dict[str, Any]]:
    """Build a ChromaDB metadata filter. Unknown ratings/costs are stored as -1 and excluded."""
    conditions: List[Dict[str, Any]] = []
    if filters.cuisines:
        clauses = [{"cuisines": {"$contains": c}} for c in filters.cuisines]
        conditions.append(clauses[0] if len(clauses) == 1 else {"$or": clauses})
    if filters.locations:
        conditions.append({"location": {"$in": filters.locations}})
    if filters.max_cost is not None:
        conditions.append({"cost_for_two": {"$lte": filters.max_cost}})
        conditions.append({"cost_for_two": {"$gte": 0}})
    if filters.min_rating is not None:
        conditions.append({"rating": {"$gte": filters.min_rating}})
    if filters.pure_veg:
        conditions.append({"pure_veg": True})
    if filters.online_order:
        conditions.append({"online_order": True})
    if filters.book_table:
        conditions.append({"book_table": True})

    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}
