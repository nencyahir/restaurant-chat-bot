import pandas as pd
import pytest

from restaurant_bot import data_loader, documents
from tests.conftest import FIXTURE_CSV


@pytest.mark.parametrize("raw, expected", [
    ("4.1/5", 4.1), ("3.9 /5", 3.9), ("NEW", None), ("-", None), (None, None), (float("nan"), None),
])
def test_parse_rating(raw, expected):
    assert data_loader.parse_rating(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("800", 800), ("1,200", 1200), (None, None), (float("nan"), None), ("abc", None),
])
def test_parse_cost(raw, expected):
    assert data_loader.parse_cost(raw) == expected


def test_parse_reviews_strips_prefix_and_bad_input():
    raw = "[('Rated 4.0', 'RATED\\n  Great food, went with my family and loved it.'), ('Rated 1.0', 'RATED\\n  ok')]"
    assert data_loader.parse_reviews(raw) == ["Great food, went with my family and loved it."]
    assert data_loader.parse_reviews("not a list") == []
    assert data_loader.parse_reviews("[broken") == []


def test_fix_text_repairs_mojibake():
    assert data_loader.fix_text("CafÃ©  Coffee\n Day") == "Café Coffee Day"


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(data_loader.DatasetError, match="not found"):
        data_loader.load_raw(tmp_path / "nope.csv")


def test_load_raw_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["x"], "address": ["y"]}).to_csv(bad, index=False)
    with pytest.raises(data_loader.DatasetError, match="missing expected columns"):
        data_loader.load_raw(bad)


def test_clean_deduplicates_listings(cleaned_df):
    raw = data_loader.load_raw(FIXTURE_CSV)
    assert len(cleaned_df) < len(raw)
    assert cleaned_df["restaurant_id"].is_unique
    jalsa = cleaned_df[cleaned_df["name"] == "Jalsa"]
    assert len(jalsa) == 1
    assert set(jalsa.iloc[0]["categories"]) == {"Buffet", "Delivery"}


def test_clean_normalises_fields(cleaned_df):
    jalsa = cleaned_df[cleaned_df["name"] == "Jalsa"].iloc[0]
    assert jalsa["rating"] == 4.1
    assert jalsa["cost_for_two"] == 800
    assert "North Indian" in jalsa["cuisines"]
    assert jalsa["online_order"] and jalsa["book_table"]
    assert cleaned_df["rating"].isna().any()  # 'NEW' restaurants have no numeric rating


def test_pure_veg_flag(cleaned_df):
    flags = dict(zip(cleaned_df["name"], cleaned_df["pure_veg"]))
    assert flags["Brundhavana Pure Veg"]
    assert not flags["Jalsa"]


def test_chunks_have_profile_and_valid_metadata(cleaned_df):
    records = cleaned_df.to_dict("records")
    chunks = documents.build_chunks(records)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert sum(c.metadata["chunk_type"] == "profile" for c in chunks) == len(records)
    for c in chunks:
        assert c.text.strip()
        for key, value in c.metadata.items():
            assert value is not None, key
            assert not (isinstance(value, list) and not value), key
        assert c.text.startswith(c.metadata["name"])  # each chunk is self-describing


def test_missing_values_become_sentinels():
    meta = documents.build_metadata({"restaurant_id": "x", "name": "A", "address": "B", "location": "C",
                                     "rating": float("nan"), "cost_for_two": None, "cuisines": []})
    assert meta["rating"] == -1.0 and meta["cost_for_two"] == -1
    assert "cuisines" not in meta


def test_review_chunking_respects_size():
    chunks = documents.split_reviews(["word " * 60] * 5, max_chars=700)
    assert len(chunks) > 1
    assert all(len(c) <= 700 for c in chunks)


def test_processed_roundtrip(cleaned_df, tmp_path):
    path = tmp_path / "restaurants.csv"
    data_loader.save_processed(cleaned_df, path)
    loaded = data_loader.load_processed(path)
    assert len(loaded) == len(cleaned_df)
    assert loaded.iloc[0]["cuisines"] == cleaned_df.iloc[0]["cuisines"]
