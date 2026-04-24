import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
_chunks: list[str] = []
_embeddings: np.ndarray | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def upload_pdf(file_bytes: bytes) -> int:
    global _chunks, _embeddings

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    chunk_size = 500
    overlap = 50
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap

    chunks = [c.strip() for c in chunks if c.strip()]

    model = _get_model()
    embeddings = model.encode(chunks, convert_to_numpy=True)

    _chunks = chunks
    _embeddings = embeddings

    return len(chunks)


def query(text: str, top_k: int = 3) -> list[str]:
    if not _chunks or _embeddings is None:
        return []

    model = _get_model()
    query_embedding = model.encode([text], convert_to_numpy=True)[0]

    norms = np.linalg.norm(_embeddings, axis=1) * np.linalg.norm(query_embedding)
    norms = np.where(norms == 0, 1e-10, norms)
    similarities = np.dot(_embeddings, query_embedding) / norms

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [_chunks[i] for i in top_indices]


def has_pdf() -> bool:
    return len(_chunks) > 0
