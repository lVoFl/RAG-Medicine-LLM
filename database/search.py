"""
Query the FAISS index using BAAI/bge-m3.
Usage:
    python search.py "查询文本" [--top_k 5]
"""

import argparse
import json
import numpy as np
import faiss
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import snapshot_download

INDEX_DIR = Path(__file__).parent / "faiss_index"


def load_model() -> BGEM3FlagModel:
    print("Loading BAAI/bge-m3 ...")
    model_dir = snapshot_download(
        repo_id="BAAI/bge-m3",
        ignore_patterns=["*.DS_Store", "imgs/*"]
    )
    model = BGEM3FlagModel(model_dir, use_fp16=True)
    print("Model loaded.")
    return model


def encode_query(query: str, model: BGEM3FlagModel) -> np.ndarray:
    vec = model.encode([query], max_length=8192)["dense_vecs"].astype("float32")
    return vec  # shape (1, 1024)


def search(query: str, model, index, chunks: list, top_k: int = 5) -> list[dict]:
    query_vec = encode_query(query, model)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        chunk = chunks[idx]
        results.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": chunk.get("chunk_id", idx),
            "source": chunk.get("source", ""),
            "headings": chunk.get("headings", ""),
            "content": chunk.get("content", "")[:300],
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("query", type=str, help="查询文本")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数量")
    args = parser.parse_args()

    print("Loading model ...")
    model = load_model()
    print("Loading FAISS index ...")
    print(str(INDEX_DIR / "index.faiss"))
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  index.ntotal = {index.ntotal}, chunks = {len(chunks)}\n")
    while(True):
        query = input()
        # print(f"Query: {query}\n{'='*60}")
        results = search(query, model, index, chunks, top_k=args.top_k)

        for r in results:
            print(f"[{r['rank']}] score={r['score']:.4f}  chunk_id={r['chunk_id']}")
            print(f"    来源: {r['source']}")
            print(f"    标题: {r['headings']}")
            print(f"    内容: {r['content']}")
            print()


if __name__ == "__main__":
    main()
