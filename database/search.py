"""
Hybrid retrieval + reranking:
1) Dense (FAISS) + BM25 recall with RRF
2) Cross-encoder rerank on recalled candidates
Usage:
    python search.py --top_k 5 --candidate_k 50 --rerank_top_n 20 --rrf_k 60
"""

import argparse
import json
import math
import re
from pathlib import Path

import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from huggingface_hub import snapshot_download

INDEX_DIR = Path(__file__).parent / "faiss_index"
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


def load_model() -> BGEM3FlagModel:
    print("Loading BAAI/bge-m3 ...")
    model_dir = snapshot_download(
        repo_id="BAAI/bge-m3",
        ignore_patterns=["*.DS_Store", "imgs/*"],
    )
    model = BGEM3FlagModel(model_dir, use_fp16=True)
    print("Model loaded.")
    return model


def load_reranker() -> FlagReranker:
    print("Loading reranker BAAI/bge-reranker-v2-m3 ...")
    reranker_dir = snapshot_download(
        repo_id="BAAI/bge-reranker-v2-m3",
        ignore_patterns=["*.DS_Store", "imgs/*"],
    )
    reranker = FlagReranker(reranker_dir, use_fp16=True)
    print("Reranker loaded.")
    return reranker


def encode_query(query: str, model: BGEM3FlagModel) -> np.ndarray:
    vec = model.encode([query], max_length=8192)["dense_vecs"].astype("float32")
    return vec  # shape (1, 1024)


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


def hybrid_recall(
    query: str,
    model,
    index,
    chunks: list,
    chunk_ids: np.ndarray,
    id_to_pos: dict[int, int],
    bm25_state: dict,
    top_k: int = 50,
    candidate_k: int = 50,
    rrf_k: int = 60,
) -> list[dict]:
    candidate_k = min(max(candidate_k, top_k), len(chunks))

    query_vec = encode_query(query, model)
    dense_scores, dense_indices = index.search(query_vec, candidate_k)
    sparse_scores, sparse_indices = bm25_search(query, bm25_state, candidate_k)

    # Reciprocal Rank Fusion to merge dense/sparse retrieval robustly.
    # Always fuse by faiss_id, not by positional offset.
    fused: dict[int, dict] = {}
    for rank, (idx, score) in enumerate(
        zip(dense_indices[0].tolist(), dense_scores[0].tolist()), start=1
    ):
        if idx < 0:
            continue
        faiss_id = int(idx)
        if faiss_id not in id_to_pos:
            continue
        fused.setdefault(faiss_id, {"dense_score": None, "bm25_score": None, "rrf": 0.0})
        fused[faiss_id]["dense_score"] = float(score)
        fused[faiss_id]["rrf"] += 1.0 / (rrf_k + rank)

    for rank, (idx, score) in enumerate(
        zip(sparse_indices.tolist(), sparse_scores.tolist()), start=1
    ):
        pos = int(idx)
        faiss_id = int(chunk_ids[pos])
        fused.setdefault(faiss_id, {"dense_score": None, "bm25_score": None, "rrf": 0.0})
        fused[faiss_id]["bm25_score"] = float(score)
        fused[faiss_id]["rrf"] += 1.0 / (rrf_k + rank)

    final = sorted(fused.items(), key=lambda item: item[1]["rrf"], reverse=True)[:top_k]

    results = []
    for rank, (faiss_id, fusion_item) in enumerate(final, start=1):
        pos = id_to_pos.get(int(faiss_id))
        if pos is None:
            continue
        chunk = chunks[pos]
        results.append(
            {
                "rank": rank,
                "rrf_score": float(fusion_item["rrf"]),
                "dense_score": fusion_item["dense_score"],
                "bm25_score": fusion_item["bm25_score"],
                # chunk_id is source metadata and may be non-unique; use faiss_id as unique key.
                "faiss_id": int(faiss_id),
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source", ""),
                "headings": chunk.get("headings", ""),
                "content": chunk.get("content", ""),
            }
        )
    return results


def rerank(
    query: str, candidates: list[dict], reranker: FlagReranker, top_k: int
) -> list[dict]:
    if not candidates:
        return []

    pairs = []
    for item in candidates:
        doc_text = (
            f"来源: {item.get('source', '')}\n"
            f"标题: {item.get('headings', '')}\n"
            f"内容: {item.get('content', '')}"
        )
        pairs.append([query, doc_text])

    rerank_scores = reranker.compute_score(pairs)
    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]

    merged = []
    for item, score in zip(candidates, rerank_scores):
        out = dict(item)
        out["rerank_score"] = float(score)
        merged.append(out)

    merged.sort(key=lambda x: x["rerank_score"], reverse=True)
    merged = merged[:top_k]
    for i, item in enumerate(merged, start=1):
        item["rank"] = i
        item["content"] = item.get("content", "")[:300]
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数量")
    parser.add_argument(
        "--candidate_k",
        type=int,
        default=50,
        help="每一路检索候选数量（向量/BM25）",
    )
    parser.add_argument(
        "--rerank_top_n",
        type=int,
        default=20,
        help="召回后参与重排的候选数量",
    )
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF 融合常数")
    args = parser.parse_args()

    print("Loading models ...")
    model = load_model()
    reranker = load_reranker()
    print("Loading FAISS index ...")
    print(str(INDEX_DIR / "index.faiss"))
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    chunk_ids_path = INDEX_DIR / "chunk_ids.npy"
    if chunk_ids_path.exists():
        chunk_ids = np.load(chunk_ids_path).astype("int64")
        if len(chunk_ids) != len(chunks):
            raise ValueError(
                f"Mismatch: chunk_ids={len(chunk_ids)} vs chunks={len(chunks)} at {chunk_ids_path}"
            )
    else:
        # Backward compatibility for old indexes without explicit id mapping.
        chunk_ids = np.arange(len(chunks), dtype="int64")
    id_to_pos = {int(fid): i for i, fid in enumerate(chunk_ids.tolist())}
    print(f"  index.ntotal = {index.ntotal}, chunks = {len(chunks)}\n")

    print("Building BM25 state ...")
    bm25_state = build_bm25_index(chunks)
    print("BM25 ready.\n")

    while True:
        query = input()
        recall_results = hybrid_recall(
            query,
            model,
            index,
            chunks,
            chunk_ids,
            id_to_pos,
            bm25_state,
            top_k=args.rerank_top_n,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )
        results = rerank(query, recall_results, reranker, top_k=args.top_k)

        for r in results:
            print(
                f"[{r['rank']}] rerank={r['rerank_score']:.4f} rrf={r['rrf_score']:.4f} "
                f"dense={r['dense_score']} bm25={r['bm25_score']} "
                f"faiss_id={r['faiss_id']} chunk_id={r['chunk_id']}"
            )
            print(f"    来源: {r['source']}")
            print(f"    标题: {r['headings']}")
            print(f"    内容: {r['content']}")
            print()


if __name__ == "__main__":
    main()
