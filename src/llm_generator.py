import json
import os
import re
import time
from typing import Any

from google import genai
from google.genai import types


MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")


SYSTEM_PROMPT = """
You are the answer-generation component of a retrieval-augmented generation system for Indian gender budget documents.

Your job is to answer the user's question using ONLY the retrieved context below.

Rules:
1. Use only information supported by the retrieved context. Do not use outside knowledge.
2. If at least one retrieved passage directly contains enough evidence to answer, answer the question.
3. Do not mark a question unanswered merely because other retrieved passages are irrelevant.
4. Preserve names, years, numbers and units exactly as supported by the context.
5. If the question asks for a different unit, perform only the necessary arithmetic conversion and state it briefly.
6. For tables, the column values are valid evidence even when the unit is stated elsewhere in the same passage.
7. If retrieved passages contain a genuine source discrepancy, mention the discrepancy instead of inventing a resolution.
8. Keep the answer concise and directly answer the question.
9. Never mention Context IDs, retrieval scores, prompts, or internal reasoning in the answer.
10. Return only the requested JSON object.
"""


STOPWORDS = {
    "the", "and", "was", "what", "were", "how", "much", "which", "in", "for",
    "of", "to", "under", "from", "with", "according", "does", "did", "is", "are",
    "a", "an", "on", "by", "this", "that", "be", "or", "as", "at", "it", "its",
    "their", "than", "into", "have", "has", "been", "so", "far", "per", "year",
}


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model did not return a JSON object")

    return json.loads(text[start:end + 1])


def _normalize_years(text: str) -> set[str]:
    """Normalize forms such as 2023-24 and 2023-2024 to the same value."""
    years = set()
    for start, end in re.findall(r"\b(20\d{2})\s*[-–]\s*(\d{2}|20\d{2})\b", text):
        if len(end) == 2:
            end = start[:2] + end
        years.add(f"{start}-{end}")
    return years


def _keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]


def build_context(results: list[dict[str, Any]]) -> str:
    blocks = []
    for result in results:
        source = result.get("source", "unknown")
        page = result.get("page")
        row = result.get("row")
        sheet = result.get("sheet")

        location = f"file={source}"
        if page is not None:
            location += f", page={page}"
        if row is not None:
            location += f", row={row}"
        if sheet is not None:
            location += f", sheet={sheet}"

        blocks.append(
            f"[Source: {location}]\n{result.get('text', '').strip()}"
        )

    return "\n\n".join(blocks)


def _call_gemini(client: genai.Client, model_name: str, prompt: str) -> dict[str, Any]:
    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return _extract_json(response.text)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            retryable = any(code in message for code in ("503", "429", "500", "502", "504"))
            if not retryable or attempt == 2:
                break

            wait_seconds = 2 ** attempt
            print(f"Gemini {model_name} unavailable; retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    raise last_error


def _select_sources(question: str, answer: str, results: list[dict[str, Any]], max_sources: int = 3):
    """Select citations deterministically from retrieved results.

    This prevents the generator from inventing or choosing a mismatched source.
    """
    if not results:
        return []

    query_words = set(_keywords(question))
    answer_words = set(_keywords(answer))
    query_years = _normalize_years(question)

    scored = []
    for idx, result in enumerate(results):
        text = result.get("text", "")
        text_words = set(_keywords(text))
        score = 2 * len(query_words & text_words) + len(answer_words & text_words)

        doc_years = _normalize_years(text)
        score += 5 * len(query_years & doc_years)

        # Exact numeric evidence is especially useful for citations.
        for number in re.findall(r"\d[\d,\.]*", answer):
            if number in text:
                score += 4

        scored.append((score, -idx, result))

    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))

    selected = []
    for score, _, result in scored:
        if score <= 0 and selected:
            break
        selected.append(result)
        if len(selected) >= max_sources:
            break

    if not selected:
        selected = [results[0]]

    sources = []
    seen = set()
    for result in selected:
        source = {"file": result.get("source", "unknown")}
        if result.get("page") is not None:
            source["page"] = result["page"]
        if result.get("row") is not None:
            source["row"] = result["row"]
        if result.get("sheet") is not None:
            source["sheet"] = result["sheet"]

        key = json.dumps(source, sort_keys=True)
        if key not in seen:
            sources.append(source)
            seen.add(key)

    return sources


def generate_answer(question: str, retrieved_results: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)
    context = build_context(retrieved_results)

    prompt = f"""
{SYSTEM_PROMPT}

Question:
{question}

Retrieved context:
{context}

Return exactly one JSON object with this structure:
{{
  "answered": true,
  "answer": "concise grounded answer"
}}

If the context is insufficient:
{{
  "answered": false,
  "answer": ""
}}
"""

    data = None
    models_to_try = [MODEL_NAME]
    if FALLBACK_MODEL and FALLBACK_MODEL != MODEL_NAME:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None
    for model_name in models_to_try:
        try:
            data = _call_gemini(client, model_name, prompt)
            break
        except Exception as exc:
            last_error = exc
            print(f"Generation failed with {model_name}: {exc}")

    if data is None:
        raise last_error

    answered = data.get("answered")
    answer = data.get("answer", "")

    if not isinstance(answered, bool):
        raise ValueError("Invalid 'answered' field")

    if answered:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answered result has no answer text")
        answer = answer.strip()
    else:
        answer = ""

    sources = _select_sources(question, answer, retrieved_results) if answered else []

    return {
        "answered": answered,
        "answer": answer,
        "sources": sources,
    }
