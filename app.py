"""Streamlit chat UI for the RAG restaurant recommendation bot.

Run with:  streamlit run app.py
"""

import logging

import pandas as pd
import streamlit as st

from restaurant_bot import config
from restaurant_bot.llm import LLMError, fallback_answer, llm_available, stream_answer
from restaurant_bot.query_parser import Filters
from restaurant_bot.retriever import Retriever
from restaurant_bot.vector_store import VectorStoreError

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Bangalore Restaurant Finder", page_icon="🍽️", layout="wide")

EXAMPLES = [
    "Find vegetarian restaurants",
    "Recommend restaurants under 400 for two",
    "Find highly rated Italian restaurants",
    "Which restaurants are good for families?",
    "Find cafes in Koramangala",
]


@st.cache_resource(show_spinner="Loading vector store and embedding model...")
def load_retriever() -> Retriever:
    retriever = Retriever()
    retriever.retrieve("warm up", top_k=1)  # loads the embedding model once
    return retriever


def sources_table(restaurants) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(restaurants, start=1):
        m = r.metadata
        rows.append({
            "#": i,
            "Restaurant": m["name"],
            "Location": m["location"],
            "Cuisines": m["cuisines_text"],
            "Type": m["rest_type"],
            "Cost for two (Rs.)": m["cost_for_two"] if m["cost_for_two"] >= 0 else None,
            "Rating": m["rating"] if m["rating"] >= 0 else None,
            "Votes": m["votes"],
            "Pure veg": "yes" if m["pure_veg"] else "",
            "Similarity": round(r.similarity, 3),
            "Zomato": m["url"],
        })
    return pd.DataFrame(rows)


def render_sources(restaurants, filters) -> None:
    applied = filters.describe()
    if applied:
        st.caption("Filters applied: " + ", ".join(f"**{k}** = {v}" for k, v in applied.items()))
    if not restaurants:
        return
    with st.expander(f"Sources: {len(restaurants)} restaurants retrieved from the dataset"):
        st.dataframe(
            sources_table(restaurants), hide_index=True, use_container_width=True,
            column_config={"Zomato": st.column_config.LinkColumn("Zomato", display_text="link")},
        )
        for i, r in enumerate(restaurants, start=1):
            with st.popover(f"[{i}] {r.metadata['name']}: retrieved text"):
                st.text(r.profile)
                for ev in r.evidence:
                    st.text(ev)


def sidebar_filters(vocab) -> Filters:
    st.sidebar.header("Filters (optional)")
    st.sidebar.caption("Filters in your question are detected automatically; these override them.")
    locations = st.sidebar.multiselect("Location", vocab.get("locations", []))
    cuisines = st.sidebar.multiselect("Cuisine", vocab.get("cuisines", []))
    max_cost = st.sidebar.slider("Max cost for two (Rs.)", 0, 6000, 0, step=100,
                                 help="0 = no limit")
    min_rating = st.sidebar.slider("Minimum rating", 0.0, 5.0, 0.0, step=0.1, help="0 = any")
    pure_veg = st.sidebar.checkbox("Pure vegetarian only")
    online_order = st.sidebar.checkbox("Online ordering available")
    book_table = st.sidebar.checkbox("Table booking available")
    st.session_state.top_k = st.sidebar.slider("Number of results", 3, 10, config.DEFAULT_TOP_K)
    return Filters(
        cuisines=cuisines, locations=locations,
        max_cost=max_cost or None, min_rating=min_rating or None,
        pure_veg=pure_veg, online_order=online_order, book_table=book_table,
    )


def main() -> None:
    st.title("🍽️ Bangalore Restaurant Finder")
    st.caption("RAG chatbot over the Zomato Bangalore dataset: answers come only from retrieved restaurant records.")

    try:
        retriever = load_retriever()
    except VectorStoreError as exc:
        st.error(f"{exc}")
        st.code("python scripts/download_data.py\npython scripts/ingest.py", language="bash")
        st.stop()

    overrides = sidebar_filters(retriever.vocab)
    if not llm_available():
        st.sidebar.warning("No ANTHROPIC_API_KEY found: showing retrieved results without LLM summaries.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("result") is not None:
                render_sources(msg["result"].restaurants, msg["result"].filters)

    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(len(EXAMPLES))
        for col, example in zip(cols, EXAMPLES):
            if col.button(example, use_container_width=True):
                st.session_state.pending = example
                st.rerun()

    question = st.chat_input("Ask for a restaurant recommendation...") or st.session_state.pop("pending", None)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching restaurants..."):
                result = retriever.retrieve(question, top_k=st.session_state.top_k, overrides=overrides)
        except Exception as exc:  # keep the chat usable even if retrieval fails
            logging.exception("Retrieval failed")
            st.error(f"Retrieval failed: {exc}")
            return

        if not result.restaurants:
            answer = ("I couldn't find any restaurants in the dataset matching that request. "
                      "Try relaxing the filters (budget, rating, location or cuisine).")
            st.markdown(answer)
        elif llm_available():
            try:
                answer = st.write_stream(stream_answer(result))
            except LLMError as exc:
                st.warning(f"{exc} Showing retrieved results instead.")
                answer = fallback_answer(result)
                st.markdown(answer)
        else:
            answer = fallback_answer(result)
            st.markdown(answer)

        render_sources(result.restaurants, result.filters)
    st.session_state.messages.append({"role": "assistant", "content": answer, "result": result})


main()
