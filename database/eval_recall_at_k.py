"""
Evaluate retrieval recall for single-gold qrels.

qrels jsonl format (one line per query):
{"query":"CKD患者的血压控制目标","relevant_faiss_ids":[12730]}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Recall@1/5/10 for retrieval")
    parser.add_argument("--qrels_file", type=str, required=True, help="jsonl with query + relevant_faiss_ids")
    parser.add_argument("--index_dir", type=str, default="faiss_index", help="directory containing faiss files")
    parser.add_argument(
        "--method",
        type=str,
        default="hybrid_rerank",
        choices=["dense", "bm25", "hybrid", "hybrid_rerank"],
        help="retrieval backend",
    )
    parser.add_argument("--top_k", type=int, default=10, help="retrieve top-k docs per query")
    parser.add_argument("--candidate_k", type=int, default=50, help="dense/hybrid candidate count")
    parser.add_argument("--rerank_top_n", type=int, default=20, help="recall size before reranking")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF constant for hybrid")
    parser.add_argument("--report_ks", type=str, default="1,5,10", help="comma-separated ks to report")
    parser.add_argument("--output_file", type=str, default=None, help="optional per-query result jsonl")
    return parser.parse_args()


def load_qrels(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = str(item.get("query", "")).strip()
            rel = item.get("relevant_faiss_ids", [])
            if not query or not isinstance(rel, list) or len(rel) == 0:
                raise ValueError(f"Invalid qrels line {i}: need query + relevant_faiss_ids")
            rows.append(
                {
                    "query": query,
                    # keep first as canonical gold for your current setting
                    "gold_faiss_id": int(rel[0]),
                    "all_relevant": [int(x) for x in rel],
                }
            )
    return rows


def parse_report_ks(text: str, top_k: int):
    ks = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        k = int(x)
        if k <= 0:
            continue
        ks.append(min(k, top_k))
    ks = sorted(set(ks))
    return ks or [1, min(5, top_k), top_k]


def hit_at_k(pred_ids, rel_set, k: int) -> int:
    return 1 if any(x in rel_set for x in pred_ids[:k]) else 0


def main():
    args = parse_args()
    report_ks = parse_report_ks(args.report_ks, args.top_k)

    try:
        import faiss
    except Exception as exc:
        raise RuntimeError("Missing dependency: faiss") from exc

    from search import bm25_search, build_bm25_index, hybrid_recall, load_model

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_ids_path = index_dir / "chunk_ids.npy"
    if chunk_ids_path.exists():
        chunk_ids = np.load(chunk_ids_path).astype("int64")
        if len(chunk_ids) != len(chunks):
            raise ValueError("chunk_ids length mismatch with chunks")
    else:
        chunk_ids = np.arange(len(chunks), dtype="int64")
    id_to_pos = {int(fid): i for i, fid in enumerate(chunk_ids.tolist())}

    qrels = load_qrels(Path(args.qrels_file))
    model = None
    bm25_state = build_bm25_index(chunks)
    runtime = None

    if args.method in {"dense", "hybrid"}:
        model = load_model()
    elif args.method == "hybrid_rerank":
        project_root = Path(__file__).resolve().parent.parent
        model_dir = project_root / "model"
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        from qwen_service.rag_runtime import RAGRuntime  # pylint: disable=import-error

        runtime = RAGRuntime(index_dir=index_dir)

    totals = {k: 0 for k in report_ks}
    details = []

    for i, row in enumerate(qrels, 1):
        query = row["query"]
        rel_set = set(row["all_relevant"])

        if args.method == "dense":
            query_vec = model.encode([query], max_length=8192)["dense_vecs"].astype("float32")
            _, dense_indices = index.search(query_vec, args.top_k)
            pred = [int(x) for x in dense_indices[0].tolist() if int(x) >= 0]
        elif args.method == "bm25":
            _, sparse_indices = bm25_search(query, bm25_state, args.top_k)
            pred = [int(chunk_ids[int(pos)]) for pos in sparse_indices.tolist()]
        elif args.method == "hybrid":
            results = hybrid_recall(
                query=query,
                model=model,
                index=index,
                chunks=chunks,
                chunk_ids=chunk_ids,
                id_to_pos=id_to_pos,
                bm25_state=bm25_state,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
            )
            pred = [int(x["faiss_id"]) for x in results]
        else:
            _, reranked = runtime.retrieve(
                query=query,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                rerank_top_n=args.rerank_top_n,
                rrf_k=args.rrf_k,
                max_chars_per_doc=500,
            )
            pred = [int(x["faiss_id"]) for x in reranked]

        row_hits = {}
        for k in report_ks:
            h = hit_at_k(pred, rel_set, k)
            totals[k] += h
            row_hits[f"hit@{k}"] = h

        details.append(
            {
                "index": i,
                "query": query,
                "gold_faiss_id": row["gold_faiss_id"],
                "pred_topk_faiss_ids": pred,
                **row_hits,
            }
        )
        hit_msg = " ".join([f"R@{k}:{row_hits[f'hit@{k}']}" for k in report_ks])
        print(f"[{i}/{len(qrels)}] {hit_msg} | {query}")

    n = max(len(qrels), 1)
    print("\n=== Recall Summary ===")
    print(f"Method: {args.method}")
    print(f"Queries: {len(qrels)}")
    for k in report_ks:
        print(f"Recall@{k}: {totals[k] / n:.4f} ({totals[k]}/{len(qrels)})")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for row in details:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved details: {args.output_file}")


if __name__ == "__main__":
    main()
