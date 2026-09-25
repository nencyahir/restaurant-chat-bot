import sys
from pathlib import Path

import chromadb
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from restaurant_bot import data_loader, documents, ingest, vector_store  # noqa: E402

FIXTURE_CSV = ROOT / "tests" / "fixtures" / "sample_zomato.csv"


@pytest.fixture(scope="session")
def cleaned_df():
    return data_loader.clean(data_loader.load_raw(FIXTURE_CSV))


@pytest.fixture(scope="session")
def collection(cleaned_df):
    """A real (in-memory) Chroma collection built from real sample rows of the dataset."""
    client = chromadb.EphemeralClient()
    chunks = documents.build_chunks(cleaned_df.to_dict("records"))
    return vector_store.build_collection(chunks, client=client, name="test_restaurants")


@pytest.fixture(scope="session")
def vocab(cleaned_df):
    return ingest.build_vocab(cleaned_df)
