# 🍽️ Restaurant Recommendation Chatbot (RAG)

A Retrieval-Augmented Generation chatbot that recommends restaurants in Bangalore from the real
Zomato dataset. You ask in plain English, for example *"highly rated Italian restaurants in
Koramangala"*. The app retrieves matching restaurants from a ChromaDB vector store, and an LLM
(Gemini or Claude) writes an answer **using only those records**, citing them as sources.

## Demo

▶️ **[Watch the demo video](docs/demo.mp4)** (4 minutes). It walks through the dataset, example questions,
automatic filter detection, the sidebar filters and the cited sources.

## Dataset

**Zomato Bangalore Restaurants**, scraped from Zomato and originally published on Kaggle by
Himanshu Poddar. This project downloads it from a public Hugging Face mirror:
[`ManikaSaini/zomato-restaurant-recommendation`](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation).

- 51,717 rows, 17 columns (~575 MB), which collapse to **12,499 unique restaurants** after de-duplication.
- Fields used: name, address, location, restaurant type, cuisines, popular dishes, rating, votes,
  cost for two, online ordering, table booking, category and customer reviews.
- A **real sample is bundled** in `data/sample/zomato_sample.csv` (4 MB): the 500 most-voted
  restaurants, so the app runs in minutes. The full file is not committed; `scripts/download_data.py` fetches it.

## RAG architecture

```
Zomato CSV ──► Load & clean ──► Chunks ──► Embeddings ──► ChromaDB
                                          (all-MiniLM-L6-v2)   │
User question ──► Filter extraction ──► Filtered semantic search ◄┘
                  (cuisine, location,          │
                   budget, rating, veg)        ▼
                                  Top-k restaurants (profile + reviews)
                                               │
                                               ▼
                        Gemini / Claude (grounded prompt) ──► Answer + cited sources
```

| Stage | Module | What it does |
|---|---|---|
| Load & clean | `restaurant_bot/data_loader.py` | Parses ratings and costs, merges duplicate listings into one record per restaurant, selects review excerpts, and derives a `pure_veg` flag from "Pure Veg" mentions. |
| Chunking | `restaurant_bot/documents.py` | One **profile chunk** (facts) plus **review chunks** (~700 chars) per restaurant, each with full metadata. |
| Embeddings & storage | `restaurant_bot/vector_store.py` | `all-MiniLM-L6-v2` embeddings in a persistent ChromaDB collection (cosine distance). |
| Filter extraction | `restaurant_bot/query_parser.py` | Detects cuisine, location, budget ("under 500", "cheap"), rating ("highly rated", "above 4.2"), vegetarian, delivery and table booking from the question. |
| Retrieval | `restaurant_bot/retriever.py` | Filtered semantic search grouped by restaurant; "best/top rated" queries are re-ranked by rating. |
| Generation | `restaurant_bot/llm.py` | The LLM sees only the retrieved records and must cite them as `[n]`. Gemini falls back to other models if one is busy. If no LLM is available, the retrieved list is shown instead. |
| UI | `app.py` | Streamlit chat with example questions, sidebar filters, and the applied filters and sources under every answer. |

Questions without a filterable attribute, such as *"good for families"*, are answered by semantic
search over customer reviews. Sidebar filters override filters detected in the question.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/nencyahir/restaurant-chat-bot.git
cd restaurant-chat-bot
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: smaller CPU-only build
pip install -r requirements.txt
cp .env.example .env
```

Then add an API key to `.env`. Gemini has a free tier ([get a key](https://aistudio.google.com/apikey)):

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
```

To use Claude instead, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...`.

## Run

```bash
python scripts/ingest.py      # build the vector store from the bundled sample (~1 min)
streamlit run app.py          # opens http://localhost:8501
pytest                        # run the tests
```

For the full dataset (12,499 restaurants, ~30–40 min to embed on CPU):

```bash
python scripts/download_data.py
python scripts/ingest.py --full
```

## Example queries

- Find vegetarian restaurants
- Recommend restaurants under 400 for two
- Find highly rated Italian restaurants
- Which restaurants are good for families?
- Find cafes in Koramangala
- North Indian food in Indiranagar with table booking

See [FILTER_SEARCH_GUIDE.md](FILTER_SEARCH_GUIDE.md) for filter + question combinations to try.

## Project structure

```
app.py                  Streamlit chatbot UI
restaurant_bot/         config, data_loader, documents, vector_store, query_parser, retriever, llm, ingest
scripts/                download_data.py, ingest.py
tests/                  Ingestion and retrieval tests
data/sample/            Bundled 500-restaurant sample
docs/demo.mp4           Demo video
```

## Limitations

- Data is a Zomato scrape from around 2019, so ratings and prices may be outdated.
- The vegetarian flag is derived from names and reviews, so some veg restaurants may be missed.
- Each question is answered on its own; earlier chat turns are not used as context.
