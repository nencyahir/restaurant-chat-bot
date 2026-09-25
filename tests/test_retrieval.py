import chromadb
import pytest

from restaurant_bot import vector_store
from restaurant_bot.llm import build_user_message, fallback_answer
from restaurant_bot.query_parser import Filters, merge_filters, parse_query, to_where
from restaurant_bot.retriever import Retriever

CUISINES = ["Italian", "North Indian", "South Indian", "Chinese", "Cafe", "Pizza"]
LOCATIONS = ["Koramangala 5th Block", "Koramangala 7th Block", "Indiranagar", "BTM", "JP Nagar"]


# ---- query parsing ---------------------------------------------------------------------------

def test_parse_vegetarian():
    assert parse_query("Find vegetarian restaurants").pure_veg
    assert parse_query("pure veg thali").pure_veg
    assert not parse_query("best non-veg biryani").pure_veg


def test_parse_budget():
    assert parse_query("restaurants under 500").max_cost == 500
    assert parse_query("dinner below Rs. 1,200 for two").max_cost == 1200
    assert parse_query("my budget is 300").max_cost == 300
    assert parse_query("cheap eats").max_cost == 500
    assert parse_query("any restaurant").max_cost is None


def test_parse_rating():
    f = parse_query("Find highly rated Italian restaurants", CUISINES)
    assert f.cuisines == ["Italian"] and f.min_rating == 4.0 and f.sort_by_rating
    assert parse_query("places rated above 4.3").min_rating == 4.3


def test_parse_cuisine_longest_match():
    assert parse_query("good south indian breakfast", CUISINES).cuisines == ["South Indian"]
    assert parse_query("cafes nearby", CUISINES).cuisines == ["Cafe"]


def test_parse_location():
    assert parse_query("cafes in Indiranagar", CUISINES, LOCATIONS).locations == ["Indiranagar"]
    # A base area name matches every sub-location in the data.
    assert parse_query("food in koramangala", CUISINES, LOCATIONS).locations == [
        "Koramangala 5th Block", "Koramangala 7th Block"]
    assert parse_query("koramangala 5th block pubs", CUISINES, LOCATIONS).locations == ["Koramangala 5th Block"]


def test_to_where():
    assert to_where(Filters()) is None
    assert to_where(Filters(pure_veg=True)) == {"pure_veg": True}
    where = to_where(Filters(cuisines=["Italian", "Pizza"], max_cost=500))
    assert where["$and"][0] == {"$or": [{"cuisines": {"$contains": "Italian"}},
                                        {"cuisines": {"$contains": "Pizza"}}]}
    assert {"cost_for_two": {"$gte": 0}} in where["$and"]


def test_merge_filters_prefers_overrides():
    merged = merge_filters(Filters(max_cost=500, cuisines=["Chinese"]), Filters(max_cost=300))
    assert merged.max_cost == 300 and merged.cuisines == ["Chinese"]


# ---- retrieval against a real Chroma collection built from dataset rows ----------------------

@pytest.fixture(scope="module")
def retriever(collection, vocab):
    return Retriever(collection=collection, vocab=vocab)


def test_retrieve_returns_unique_restaurants_with_profiles(retriever):
    result = retriever.retrieve("good place for dinner", top_k=5)
    ids = [r.restaurant_id for r in result.restaurants]
    assert 0 < len(ids) <= 5 and len(ids) == len(set(ids))
    assert all(r.profile.startswith(r.metadata["name"]) for r in result.restaurants)


def test_retrieve_cuisine_filter(retriever):
    result = retriever.retrieve("Find highly rated Italian restaurants")
    assert result.restaurants
    for r in result.restaurants:
        assert "Italian" in r.metadata["cuisines"]
        assert r.metadata["rating"] >= 4.0
    ratings = [r.metadata["rating"] for r in result.restaurants]
    assert ratings == sorted(ratings, reverse=True)


def test_retrieve_budget_filter(retriever):
    result = retriever.retrieve("cheap food under 300")
    assert result.restaurants
    assert all(0 <= r.metadata["cost_for_two"] <= 300 for r in result.restaurants)


def test_retrieve_vegetarian_filter(retriever):
    result = retriever.retrieve("Find vegetarian restaurants")
    assert result.restaurants
    assert all(r.metadata["pure_veg"] for r in result.restaurants)


def test_retrieve_location_filter(retriever):
    result = retriever.retrieve("restaurants in Koramangala")
    assert result.restaurants
    assert all(r.metadata["location"].startswith("Koramangala") for r in result.restaurants)


def test_retrieve_semantic_family(retriever):
    result = retriever.retrieve("Which restaurants are good for families?", top_k=3)
    assert result.restaurants
    text = " ".join(r.profile + " ".join(r.evidence) for r in result.restaurants).lower()
    assert "family" in text


def test_retrieve_no_match_returns_empty(retriever):
    result = retriever.retrieve("Italian restaurants under 100")
    assert result.restaurants == []
    assert "couldn't find" in fallback_answer(result)


def test_retrieve_rejects_empty_query(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_missing_collection_raises():
    with pytest.raises(vector_store.VectorStoreError):
        vector_store.get_collection(client=chromadb.EphemeralClient(), name="does-not-exist")


def test_llm_prompt_contains_only_retrieved_records(retriever):
    result = retriever.retrieve("Find highly rated Italian restaurants", top_k=2)
    message = build_user_message(result)
    assert message.count("<record ") == len(result.restaurants)
    for r in result.restaurants:
        assert r.metadata["name"] in message
    assert all(r.metadata["name"] in fallback_answer(result) for r in result.restaurants)
