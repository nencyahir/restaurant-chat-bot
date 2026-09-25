"""Load and clean the Zomato Bangalore restaurant dataset.

The raw file has one row per (restaurant, listing category), so the same restaurant can
appear several times (e.g. under "Delivery" and "Dine-out"). Cleaning collapses those into
one record per restaurant and normalises ratings, costs and review text.
"""

import ast
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from restaurant_bot import config

logger = logging.getLogger(__name__)

RAW_COLUMNS = {
    "name": "name",
    "address": "address",
    "location": "location",
    "listed_in(city)": "city",
    "rest_type": "rest_type",
    "cuisines": "cuisines",
    "dish_liked": "dish_liked",
    "rate": "rate",
    "votes": "votes",
    "approx_cost(for two people)": "cost",
    "online_order": "online_order",
    "book_table": "book_table",
    "listed_in(type)": "category",
    "reviews_list": "reviews_list",
    "url": "url",
}

# Reviews mentioning these topics are kept first, since they answer common questions.
REVIEW_KEYWORDS = re.compile(
    r"\b(family|families|kids|children|veg|vegetarian|budget|cheap|affordable|pocket|"
    r"ambience|service|value|group|friends|date|romantic|buffet)\b",
    re.IGNORECASE,
)
PURE_VEG_PATTERN = re.compile(r"\bpure[\s-]*veg(etarian)?\b", re.IGNORECASE)
VEG_NAME_PATTERN = re.compile(r"\bveg(etarian)?\b", re.IGNORECASE)
NON_VEG_PATTERN = re.compile(r"\bnon[\s-]*veg", re.IGNORECASE)


class DatasetError(Exception):
    """Raised when the dataset is missing or does not have the expected shape."""


def fix_text(text: str) -> str:
    """Repair common mojibake (UTF-8 read as Latin-1) and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    if "Ã" in text or "â" in text:
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return re.sub(r"\s+", " ", text).strip()


def parse_rating(value) -> Optional[float]:
    """'4.1/5' or '3.9 /5' -> 4.1; 'NEW', '-', NaN -> None."""
    if not isinstance(value, str):
        return None
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*/\s*5\s*$", value)
    return float(match.group(1)) if match else None


def parse_cost(value) -> Optional[int]:
    """'1,200' -> 1200; missing or malformed -> None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    digits = str(value).replace(",", "").strip()
    return int(digits) if digits.isdigit() else None


def split_list(value) -> List[str]:
    """'North Indian, Chinese' -> ['North Indian', 'Chinese']."""
    if not isinstance(value, str):
        return []
    items = [fix_text(item) for item in value.split(",")]
    return list(dict.fromkeys(item for item in items if item))


def parse_reviews(raw) -> List[str]:
    """Parse the stringified list of (rating, 'RATED\\n text') tuples into clean review texts."""
    if not isinstance(raw, str) or not raw.startswith("["):
        return []
    try:
        items = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    reviews = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        text = fix_text(re.sub(r"^\s*RATED\s*", "", str(item[1])))
        if len(text) >= 30:
            reviews.append(text)
    return list(dict.fromkeys(reviews))


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "..."


def select_reviews(reviews: List[str]) -> List[str]:
    """Keep a handful of short review excerpts, preferring ones that mention useful topics."""
    ranked = sorted(reviews, key=lambda r: 0 if REVIEW_KEYWORDS.search(r) else 1)
    return [truncate(r, config.MAX_REVIEW_CHARS) for r in ranked[: config.MAX_REVIEWS_PER_RESTAURANT]]


def is_pure_veg(name: str, reviews: List[str]) -> bool:
    """Flag restaurants that the data itself describes as vegetarian ('Pure Veg' etc.)."""
    if VEG_NAME_PATTERN.search(name) and not NON_VEG_PATTERN.search(name):
        return True
    return any(PURE_VEG_PATTERN.search(review) for review in reviews)


def restaurant_id(name: str, address: str) -> str:
    return hashlib.md5(f"{name}|{address}".encode("utf-8")).hexdigest()[:16]


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS, dtype=str)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise DatasetError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df.rename(columns=RAW_COLUMNS)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise fields and collapse duplicate listings into one row per restaurant."""
    df = df.copy()
    df["name"] = df["name"].map(fix_text)
    df["address"] = df["address"].map(fix_text)
    df = df[(df["name"] != "") & (df["address"] != "")]

    df["rating"] = df["rate"].map(parse_rating)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    df["cost_for_two"] = df["cost"].map(parse_cost)
    df["review_len"] = df["reviews_list"].fillna("").str.len()

    # Every category a restaurant is listed under (Delivery, Dine-out, Buffet, ...).
    categories = (
        df.groupby(["name", "address"])["category"]
        .agg(lambda s: sorted(set(s.dropna())))
        .rename("categories")
    )

    # Keep the most informative listing per restaurant (most votes, then most reviews).
    df = df.sort_values(["votes", "review_len"], ascending=False)
    df = df.drop_duplicates(["name", "address"], keep="first")
    df = df.join(categories, on=["name", "address"])

    records = []
    for row in df.itertuples(index=False):
        reviews = parse_reviews(row.reviews_list)
        records.append(
            {
                "restaurant_id": restaurant_id(row.name, row.address),
                "name": row.name,
                "address": row.address,
                "location": fix_text(row.location) or fix_text(row.city),
                "city": fix_text(row.city),
                "rest_type": fix_text(row.rest_type),
                "cuisines": split_list(row.cuisines),
                "dishes_liked": split_list(row.dish_liked),
                "rating": row.rating,
                "votes": row.votes,
                "cost_for_two": row.cost_for_two,
                "online_order": row.online_order == "Yes",
                "book_table": row.book_table == "Yes",
                "categories": row.categories,
                "pure_veg": is_pure_veg(row.name, reviews),
                "reviews": select_reviews(reviews),
                "url": str(row.url).split("?")[0],
            }
        )
    cleaned = pd.DataFrame.from_records(records)
    logger.info("Cleaned dataset: %d unique restaurants", len(cleaned))
    return cleaned.reset_index(drop=True)


LIST_COLUMNS = ["cuisines", "dishes_liked", "categories", "reviews"]


def save_processed(df: pd.DataFrame, path: Path = config.PROCESSED_DATA_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in LIST_COLUMNS:
        out[col] = out[col].map(json.dumps)
    out.to_csv(path, index=False)


def load_processed(path: Path = config.PROCESSED_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Processed data not found at {path}. Run `python scripts/ingest.py`.")
    df = pd.read_csv(path)
    for col in LIST_COLUMNS:
        df[col] = df[col].map(json.loads)
    df["rating"] = df["rating"].astype(object).where(df["rating"].notna(), None)
    df["cost_for_two"] = df["cost_for_two"].astype(object).where(df["cost_for_two"].notna(), None)
    return df
