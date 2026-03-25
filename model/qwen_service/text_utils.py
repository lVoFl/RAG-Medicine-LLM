import math
import re

import numpy as np


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


def clip_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def chunk_text(chunk: dict) -> str:
    parts = [
        str(chunk.get("source", "")),
        str(chunk.get("headings", "")),
        str(chunk.get("content", "")),
    ]
    return " ".join(parts)


def build_bm25_index(chunks: list[dict]) -> dict:
    doc_tf: list[dict[str, int]] = []
    doc_lens: list[int] = []
    df: dict[str, int] = {}

    for chunk in chunks:
        tokens = tokenize(chunk_text(chunk))
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        doc_tf.append(tf)
        doc_lens.append(len(tokens))
        for token in tf.keys():
            df[token] = df.get(token, 0) + 1

    n_docs = len(chunks)
    avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0
    idf = {
        token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in df.items()
    }

    return {
        "doc_tf": doc_tf,
        "doc_lens": doc_lens,
        "avgdl": avgdl,
        "idf": idf,
    }


def bm25_search(
    query: str,
    bm25_state: dict,
    top_n: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    query_terms = tokenize(query)
    doc_tf = bm25_state["doc_tf"]
    doc_lens = bm25_state["doc_lens"]
    avgdl = bm25_state["avgdl"] or 1.0
    idf = bm25_state["idf"]
    scores = np.zeros(len(doc_tf), dtype="float32")

    for i, tf in enumerate(doc_tf):
        dl = doc_lens[i]
        norm = k1 * (1 - b + b * dl / avgdl)
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            score += idf.get(term, 0.0) * (f * (k1 + 1)) / (f + norm)
        scores[i] = score

    if len(scores) == 0:
        return np.array([], dtype="float32"), np.array([], dtype="int64")

    top_n = max(1, min(top_n, len(scores)))
    indices = np.argpartition(scores, -top_n)[-top_n:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return scores[indices], indices.astype("int64")

