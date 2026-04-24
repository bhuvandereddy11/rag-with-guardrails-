import os
import json
from groq import Groq

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _client


def evaluate_rag(query: str, answer: str, context_chunks: list[str]) -> dict:
    client = _get_client()

    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = """You are an expert RAG evaluation judge. Score the answer on 3 metrics, each from 0.0 to 1.0:

- faithfulness: Is every claim in the answer supported by the retrieved context? Penalize hallucinations heavily. 1.0 = fully supported, 0.0 = completely fabricated.
- answer_relevance: Does the answer directly address the question asked? 1.0 = perfectly relevant, 0.0 = completely off-topic.
- context_precision: What fraction of the retrieved chunks were actually useful for answering? 1.0 = all chunks relevant, 0.0 = no chunks useful.

IMPORTANT: You must carefully read the question, context, and answer and give accurate scores. Do NOT return the same scores every time. Scores should reflect actual quality - a perfect answer gets 0.95-1.0, a partial answer gets 0.5-0.7, a wrong answer gets 0.1-0.3.

Return ONLY valid JSON with no markdown fences. Format: {"faithfulness": <float>, "answer_relevance": <float>, "context_precision": <float>, "feedback": "<one sentence>"}"""

    user_message = f"""QUESTION: {query}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

Analyze the above carefully and return your scores as JSON."""

    print(f"[Evaluator] evaluate_rag called. Query: {query[:80]}...")

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=256,
            temperature=0.1,
            timeout=10,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        raw = response.choices[0].message.content
        print(f"[Evaluator] Raw Groq response: {repr(raw)}")

        raw_stripped = raw.strip()
        if raw_stripped.startswith("```"):
            lines = raw_stripped.split("\n")
            raw_stripped = "\n".join(lines[1:-1]) if len(lines) > 2 else raw_stripped

        result = json.loads(raw_stripped)
        faithfulness = float(result.get("faithfulness", 0.5))
        answer_relevance = float(result.get("answer_relevance", 0.5))
        context_precision = float(result.get("context_precision", 0.5))
        result["overall"] = round((faithfulness + answer_relevance + context_precision) / 3, 4)
        return result

    except Exception as e:
        print(f"[Evaluator] Error during evaluation: {e}")
        return {
            "faithfulness": 0.5,
            "answer_relevance": 0.5,
            "context_precision": 0.5,
            "overall": 0.5,
            "feedback": f"Error: {e}",
        }
