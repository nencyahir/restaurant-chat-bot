"""Grounded answer generation with Claude, plus a no-LLM fallback."""

import logging
import os
from typing import Iterator, List

import anthropic

from restaurant_bot import config
from restaurant_bot.retriever import RetrievalResult, RetrievedRestaurant

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a restaurant recommendation assistant for Bangalore, India.

You answer using ONLY the restaurant records provided inside <restaurants> in the user's message. \
These records come from a Zomato dataset and are the only source of truth.

Rules:
- Recommend only restaurants that appear in the records. Never invent restaurants, dishes, prices, \
ratings, addresses, or features.
- Every fact you state must be supported by a record. Cite the record number in square brackets, e.g. [2].
- If the records only partially match the request (e.g. nothing explicitly mentions families), say so \
plainly and explain what the records do show.
- If no record fits, say you could not find a matching restaurant in the dataset. Do not use outside knowledge.
- Costs are approximate cost for two people in Indian Rupees (Rs.). Ratings are out of 5.
- Be concise: a one-line intro, then a short bulleted list (name, area, cuisine, cost for two, rating, \
and why it fits), then nothing else."""


class LLMError(Exception):
    """Raised when the LLM call fails in a way the user should see."""


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def format_context(restaurants: List[RetrievedRestaurant]) -> str:
    blocks = []
    for i, r in enumerate(restaurants, start=1):
        parts = [r.profile] + r.evidence
        blocks.append(f'<record number="{i}">\n' + "\n".join(p for p in parts if p) + "\n</record>")
    return "<restaurants>\n" + "\n".join(blocks) + "\n</restaurants>"


def build_user_message(result: RetrievalResult) -> str:
    applied = result.filters.describe()
    filter_note = f"Filters already applied to the search: {applied}\n\n" if applied else ""
    return f"{format_context(result.restaurants)}\n\n{filter_note}Question: {result.query}"


def stream_answer(result: RetrievalResult, client: anthropic.Anthropic = None) -> Iterator[str]:
    """Stream Claude's grounded answer as text chunks."""
    client = client or anthropic.Anthropic()
    try:
        with client.beta.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(result)}],
            output_config={"effort": "medium"},
            # If a safety classifier declines, the API retries on a recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n_The model declined to answer this request._"
        elif final.stop_reason == "max_tokens":
            yield "\n\n_(Answer truncated.)_"
    except anthropic.AuthenticationError as exc:
        raise LLMError("Invalid Anthropic API key. Check ANTHROPIC_API_KEY in your .env file.") from exc
    except anthropic.RateLimitError as exc:
        raise LLMError("Rate limited by the Anthropic API. Please wait a moment and retry.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError("Could not reach the Anthropic API. Check your internet connection.") from exc


def _fmt_cost(meta) -> str:
    return f"Rs. {meta['cost_for_two']} for two" if meta.get("cost_for_two", -1) >= 0 else "cost not listed"


def _fmt_rating(meta) -> str:
    return f"{meta['rating']}/5 ({meta['votes']} votes)" if meta.get("rating", -1) >= 0 else "not yet rated"


def fallback_answer(result: RetrievalResult) -> str:
    """Deterministic answer built directly from retrieved records (used when no LLM is configured)."""
    if not result.restaurants:
        return "I couldn't find any restaurants in the dataset matching that request."
    lines = ["Here are the best matches from the dataset:\n"]
    for i, r in enumerate(result.restaurants, start=1):
        m = r.metadata
        lines.append(
            f"- **{m['name']}** [{i}]: {m['rest_type']} in {m['location']} · {m['cuisines_text']} · "
            f"{_fmt_cost(m)} · {_fmt_rating(m)}"
        )
    return "\n".join(lines)
