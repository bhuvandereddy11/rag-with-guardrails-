import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from groq import Groq
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

import rag
import guardrails
import evaluator

app = FastAPI(title="RAG Eval System")

_history: list[dict] = []
_flagged_count = 0
_rag_count = 0
_blocked_count = 0

_groq_client = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _groq_client


class ChatRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    file_bytes = await file.read()
    chunk_count = rag.upload_pdf(file_bytes)
    print(f"[Main] PDF uploaded: {file.filename}, chunks: {chunk_count}")
    return {"status": "ok", "chunks": chunk_count}


@app.post("/chat")
async def chat(request: ChatRequest):
    global _history, _flagged_count, _rag_count, _blocked_count

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    print(f"\n[Main] /chat called. Query: {query}")

    # Step 1: Input guardrail check
    input_check = guardrails.check_input(query)
    print(f"[Main] Input check result: {input_check}")

    if not input_check.get("safe", True):
        _blocked_count += 1
        _flagged_count += 1
        reason = input_check.get("reason", "Flagged by guardrails")
        flagged_type = input_check.get("flagged_type", "unknown")

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "answer": f"Blocked: {reason}",
            "path": "blocked",
            "flagged": True,
            "flagged_type": flagged_type,
            "grounded": None,
            "eval_scores": None,
        }
        _history.append(entry)
        if len(_history) > 50:
            _history = _history[-50:]

        return {
            "answer": f"Blocked: {reason}",
            "path": "blocked",
            "confidence": 100,
            "reason": flagged_type,
            "eval_scores": None,
            "grounded": None,
        }

    # Step 2: Retrieve context chunks
    chunks = rag.query(query)
    print(f"[Main] Retrieved {len(chunks)} chunks")

    # Step 3: Generate answer with Claude
    context_text = "\n\n---\n\n".join(chunks) if chunks else "No document context available."

    client = _get_groq_client()

    rag_system_prompt = """You are a helpful document assistant. Answer the user's question based ONLY on the provided context.
If the context does not contain enough information to answer, say so clearly.
Do not make up information not present in the context. Be concise and direct."""

    rag_user_message = f"""CONTEXT:
{context_text}

QUESTION: {query}

Answer based only on the context above."""

    print(f"[Main] Calling Groq for answer generation...")
    rag_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        timeout=10,
        messages=[
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": rag_user_message},
        ],
    )
    answer = rag_response.choices[0].message.content
    print(f"[Main] Groq answer: {answer[:200]}...")

    # Step 4: Output guardrail check
    output_check = guardrails.check_output(answer, chunks)
    print(f"[Main] Output check result: {output_check}")
    grounded = output_check.get("grounded", True)

    # Step 5: Evaluate RAG quality
    eval_scores = evaluator.evaluate_rag(query, answer, chunks)
    print(f"[Main] Eval scores: {eval_scores}")

    # Step 6: Save to history
    _rag_count += 1
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": answer,
        "path": "rag",
        "flagged": False,
        "flagged_type": None,
        "grounded": grounded,
        "eval_scores": eval_scores,
    }
    _history.append(entry)
    if len(_history) > 50:
        _history = _history[-50:]

    return {
        "answer": answer,
        "path": "rag",
        "confidence": int(output_check.get("confidence", 0.8) * 100),
        "reason": "Retrieved from document context",
        "eval_scores": eval_scores,
        "grounded": grounded,
    }


@app.get("/eval-dashboard")
async def eval_dashboard():
    entries = list(_history)

    rag_entries = [e for e in entries if e["path"] == "rag" and e.get("eval_scores")]

    def safe_avg(values):
        valid = [v for v in values if v is not None]
        return round(sum(valid) / len(valid), 4) if valid else 0.0

    avg_faithfulness = safe_avg([e["eval_scores"].get("faithfulness") for e in rag_entries])
    avg_relevance = safe_avg([e["eval_scores"].get("answer_relevance") for e in rag_entries])
    avg_precision = safe_avg([e["eval_scores"].get("context_precision") for e in rag_entries])

    flagged_count = sum(1 for e in entries if e.get("flagged"))
    ungrounded_count = sum(1 for e in entries if e.get("grounded") is False)

    return {
        "entries": entries,
        "stats": {
            "total": len(entries),
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "avg_precision": avg_precision,
            "flagged_count": flagged_count,
            "ungrounded_count": ungrounded_count,
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "pdf_loaded": rag.has_pdf()}


app.mount("/static", StaticFiles(directory="static"), name="static")
