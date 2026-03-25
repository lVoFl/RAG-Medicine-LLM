import json
import threading
import time
from pathlib import Path

import faiss
import numpy as np
from huggingface_hub import snapshot_download

from qwen_service.text_utils import bm25_search, build_bm25_index, clip_text


class RAGRuntime:
    def __init__(self, index_dir: Path, debug_log: bool = False, debug_max_chars: int = 800):
        self.index_dir = index_dir
        self.lock = threading.Lock()
        self.debug_log = debug_log
        self.debug_max_chars = debug_max_chars
        # Lazy import to avoid loading heavy sklearn/scipy deps when RAG is disabled.
        try:
            from FlagEmbedding import BGEM3FlagModel, FlagReranker  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Failed to import FlagEmbedding for RAG. "
                "Please check FlagEmbedding/transformers/scikit-learn/scipy compatibility "
                "in your current environment."
            ) from exc

        print("Loading BAAI/bge-m3 ...")
        model_dir = snapshot_download(
            repo_id="BAAI/bge-m3",
            ignore_patterns=["*.DS_Store", "imgs/*"],
        )
        self.model = BGEM3FlagModel(model_dir, use_fp16=True)

        print("Loading reranker BAAI/bge-reranker-v2-m3 ...")
        reranker_dir = snapshot_download(
            repo_id="BAAI/bge-reranker-v2-m3",
            ignore_patterns=["*.DS_Store", "imgs/*"],
        )
        self.reranker = FlagReranker(reranker_dir, use_fp16=True)

        index_path = self.index_dir / "index.faiss"
        chunks_path = self.index_dir / "chunks.json"
        print(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"Loaded chunks: {len(self.chunks)}")

        print("Building BM25 state ...")
        self.bm25_state = build_bm25_index(self.chunks)
        print("RAG runtime is ready.")

    def _log(self, msg: str):
        if self.debug_log:
            print(f"[RAG] {msg}")

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], max_length=8192)["dense_vecs"].astype("float32")
        return vec

    def hybrid_recall(
        self,
        query: str,
        top_k: int = 20,
        candidate_k: int = 50,
        rrf_k: int = 60,
    ) -> list[dict]:
        candidate_k = min(max(candidate_k, top_k), len(self.chunks))
        query_vec = self.encode_query(query)

        dense_scores, dense_indices = self.index.search(query_vec, candidate_k)
        sparse_scores, sparse_indices = bm25_search(query, self.bm25_state, candidate_k)

        fused: dict[int, dict] = {}
        for rank, (idx, score) in enumerate(
            zip(dense_indices[0].tolist(), dense_scores[0].tolist()), start=1
        ):
            if idx < 0:
                continue
            fused.setdefault(idx, {"dense_score": None, "bm25_score": None, "rrf": 0.0})
            fused[idx]["dense_score"] = float(score)
            fused[idx]["rrf"] += 1.0 / (rrf_k + rank)

        for rank, (idx, score) in enumerate(
            zip(sparse_indices.tolist(), sparse_scores.tolist()), start=1
        ):
            fused.setdefault(idx, {"dense_score": None, "bm25_score": None, "rrf": 0.0})
            fused[idx]["bm25_score"] = float(score)
            fused[idx]["rrf"] += 1.0 / (rrf_k + rank)

        final = sorted(fused.items(), key=lambda item: item[1]["rrf"], reverse=True)[:top_k]
        results = []
        for rank, (idx, fusion_item) in enumerate(final, start=1):
            chunk = self.chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "rrf_score": float(fusion_item["rrf"]),
                    "dense_score": fusion_item["dense_score"],
                    "bm25_score": fusion_item["bm25_score"],
                    "chunk_id": chunk.get("chunk_id", idx),
                    "source": chunk.get("source", ""),
                    "headings": chunk.get("headings", ""),
                    "content": chunk.get("content", ""),
                }
            )
        return results

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
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

        rerank_scores = self.reranker.compute_score(pairs)
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
        return merged

    def build_context(self, retrieved_docs: list[dict], max_chars_per_doc: int = 500) -> str:
        lines = []
        for i, item in enumerate(retrieved_docs, start=1):
            content = (item.get("content") or "")[:max_chars_per_doc]
            lines.append(
                f"[{i}] 来源: {item.get('source', '')}\n"
                f"标题: {item.get('headings', '')}\n"
                f"内容: {content}"
            )
        return "\n\n".join(lines)

    def retrieve(
        self,
        query: str,
        top_k: int,
        candidate_k: int,
        rerank_top_n: int,
        rrf_k: int,
        max_chars_per_doc: int,
    ) -> tuple[str, list[dict]]:
        start_t = time.time()
        self._log(
            "retrieve start | "
            f"query='{clip_text(query, self.debug_max_chars)}' top_k={top_k} "
            f"candidate_k={candidate_k} rerank_top_n={rerank_top_n} rrf_k={rrf_k}"
        )
        with self.lock:
            recall_results = self.hybrid_recall(
                query,
                top_k=rerank_top_n,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
            )
            reranked = self.rerank(query, recall_results, top_k=top_k)

        if reranked:
            for item in reranked:
                self._log(
                    "doc "
                    f"rank={item.get('rank')} "
                    f"rerank={item.get('rerank_score')} "
                    f"rrf={item.get('rrf_score')} "
                    f"source='{clip_text(str(item.get('source', '')), 120)}' "
                    f"headings='{clip_text(str(item.get('headings', '')), 120)}' "
                    f"content='{clip_text(str(item.get('content', '')), self.debug_max_chars)}'"
                )
        else:
            self._log("no retrieved docs")

        context = self.build_context(reranked, max_chars_per_doc=max_chars_per_doc)
        self._log(
            f"retrieve done | docs={len(reranked)} "
            f"context_chars={len(context)} "
            f"elapsed_ms={int((time.time() - start_t) * 1000)}"
        )
        return context, reranked

