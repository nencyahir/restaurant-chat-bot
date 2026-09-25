# 🍽️ Restaurant Recommendation Chatbot (RAG)

A Retrieval-Augmented Generation chatbot that recommends restaurants in Bangalore from a real
third-party dataset. You ask questions in plain English, for example *"highly rated Italian
restaurants in Koramangala"*. The app retrieves matching restaurant records from a ChromaDB vector
store, and Claude writes an answer **using only those records**, citing them as sources.

## Dataset

**Zomato Bangalore Restaurants.** The dataset was scraped from Zomato and originally published on
Kaggle by Himanshu Poddar. This project downloads a public mirror from Hugging Face:
[`ManikaSaini/zomato-restaurant-recommendation`](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation).

- 51,717 rows and 17 columns (~575 MB). One row exists per *(restaurant, listing category)*.
- After de-duplication by name and address: **12,499 unique restaurants**.
- Columns used: `name`, `address`, `location`, `listed_in(city)`, `rest_type`, `cuisines`,
  `dish_liked`, `rate`, `votes`, `approx_cost(for two people)`, `online_order`, `book_table`,
  `listed_in(type)`, `reviews_list`, `url`.

A **small sample is bundled** in `data/sample/zomato_sample.csv` (4 MB): all 3,806 listing rows of
the 500 most-voted restaurants, with reviews trimmed. It is real data from the same dataset and lets
the app run in minutes. The full raw file is **not committed**; `scripts/download_data.py` fetches it.

## RAG architecture

```
Zomato CSV ──► Load & clean ──► Documents ──► Chunks ──► Embeddings ──► ChromaDB
                                                     (all-MiniLM-L6-v2)      │
User question ──► Filter extraction ──► Filtered semantic search ◄───────────┘
                  (cuisine, location,         │
                   budget, rating, veg)       ▼
                                  Top-k restaurants (profile + review evidence)
                                              │
                                              ▼
                                Claude (grounded prompt) ──► Answer + cited sources
```

| Stage | Module | What it does |
|---|---|---|
| Load & clean | `restaurant_bot/data_loader.py` | Parses ratings (`"4.1/5"`, `"NEW"`) and costs (`"1,200"`). Fixes mojibake. Collapses duplicate listings into one record per restaurant and keeps all categories. Parses and selects review excerpts. Derives a `pure_veg` flag from explicit "Pure Veg" mentions in names and reviews. |
| Documents & chunking | `restaurant_bot/documents.py` | Each restaurant produces one **profile chunk** (facts) plus **review chunks** of about 700 characters. Every chunk starts with a name/location/cuisine header and carries the full metadata. |
| Embeddings & storage | `restaurant_bot/vector_store.py` | `sentence-transformers/all-MiniLM-L6-v2` (normalized) stored in a persistent ChromaDB collection with cosine distance. |
| Filter extraction | `restaurant_bot/query_parser.py` | Rule-based detection of cuisine and location (matched against the dataset's own vocabulary), budget ("under 500", "cheap"), rating ("highly rated", "above 4.2"), vegetarian, delivery and table booking. These become a ChromaDB `where` clause. Sidebar filters override them. |
| Retrieval | `restaurant_bot/retriever.py` | Filtered semantic search that over-fetches chunks, groups them by restaurant, and always attaches each restaurant's profile. For "best/top rated" queries it re-ranks by rating. |
| Generation | `restaurant_bot/llm.py` | Claude (`claude-opus-5-5`) receives only the numbered retrieved records and must cite them as `[n]`. If nothing matches, it says so. Without an API key, the app shows a deterministic list of the retrieved records instead. |
| UI | `app.py` | Streamlit chat with example questions, optional sidebar filters, and the applied filters and source restaurants shown under every answer. |

**Metadata filters supported:** cuisine (exact list match), location (an area such as "Koramangala"
matches all its blocks), max cost for two, min rating, pure veg, online ordering, table booking.
Questions without an explicit attribute, such as "good for families", are answered by semantic
search over the review chunks.

## Project structure

```
app.py                     Streamlit chatbot UI
restaurant_bot/            config, data_loader, documents, vector_store,
                           query_parser, retriever, llm, ingest
scripts/download_data.py   Download the dataset
scripts/ingest.py          Build the ChromaDB vector store
tests/                     Ingestion and retrieval tests (fixture = 11 real dataset rows)
```

## Setup

Requires Python 3.10+.

```bash
git clone git@github.com:nencyahir/restaurant-chat-bot.git
cd restaurant-chat-bot
python3 -m venv .venv && source .venv/bin/activate
# Optional, recommended on machines without a GPU: the CPU-only PyTorch build is much smaller
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env        # then put your ANTHROPIC_API_KEY in .env
```

## Run

```bash
python scripts/ingest.py               # bundled 500-restaurant sample (~1 min)
streamlit run app.py                   # opens http://localhost:8501
```

To use the full dataset (12,499 restaurants, ~34k chunks, about 30-40 min to embed on CPU):

```bash
python scripts/download_data.py        # ~575 MB, one time
python scripts/ingest.py --full        # add --limit 2000 for a faster partial build
```

Run the tests:

```bash
pytest
```

## Example queries

- Find vegetarian restaurants
- Recommend restaurants under 400 for two
- Find highly rated Italian restaurants
- Which restaurants are good for families?
- Find cafes in Koramangala
- North Indian food in Indiranagar with table booking
- Cheap South Indian breakfast in Jayanagar rated above 4

## Notes and limitations

- Answers are limited to the dataset, which is a Zomato scrape of Bangalore restaurants from around 2019.
  Ratings and prices may have changed since then.
- The `pure_veg` flag is derived from explicit mentions in restaurant names and reviews, because the
  dataset has no vegetarian column. Some vegetarian restaurants may be missed.
- Each question is answered on its own; the chatbot does not use earlier turns as context.
- API keys live in `.env`, which is git-ignored.
