"""
Append one plain-text document to FAISS index.

Flow:
1) read text file
2) split text into chunks
3) embed chunks with BAAI/bge-m3
4) append vectors to index.faiss
5) append chunk metadata to chunks.json
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel


def parse_args():
    parser = argparse.ArgumentParser(description="Append text chunks into FAISS index.")
    parser.add_argument("--text-file", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--headings", type=str, default="")
    parser.add_argument("--category", type=str, default="")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--index-dir", type=str, default=str(Path(__file__).parent / "faiss_index"))
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=8192)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", normalized) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if len(para) <= max_chars:
            buffer = para
            continue
        start = 0
        while start < len(para):
            end = min(start + max_chars, len(para))
            chunks.append(para[start:end].strip())
            if end >= len(para):
                break
            start = max(0, end - overlap)
        buffer = ""
    if buffer:
        chunks.append(buffer)

    return [c for c in chunks if c]


def main():
    args = parse_args()
    t0 = time.time()
    text_file = Path(args.text_file)
    index_dir = Path(args.index_dir)
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.json"
    chunk_ids_path = index_dir / "chunk_ids.npy"

    print(
        json.dumps(
            {
                "stage": "start",
                "text_file": str(text_file),
                "index_dir": str(index_dir),
                "source": args.source,
                "max_chars": args.max_chars,
                "overlap": args.overlap,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    raw_text = text_file.read_text(encoding="utf-8")
    chunk_texts = split_text(raw_text, max_chars=args.max_chars, overlap=args.overlap)
    if not chunk_texts:
        raise RuntimeError("text is empty after normalization")
    print(json.dumps({"stage": "split", "chunks": len(chunk_texts)}, ensure_ascii=False), flush=True)

    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        if not isinstance(chunks, list):
            raise RuntimeError("chunks.json format invalid")
    else:
        chunks = []

    max_chunk_id = -1
    for item in chunks:
        try:
            max_chunk_id = max(max_chunk_id, int(item.get("chunk_id", -1)))
        except Exception:
            continue
    next_chunk_id = max_chunk_id + 1

    new_chunks = []
    for i, content in enumerate(chunk_texts):
        new_chunks.append(
            {
                "chunk_id": next_chunk_id + i,
                "source": args.source.strip(),
                "headings": (args.headings or args.source).strip(),
                "content": content,
                "category": args.category.strip(),
                "version": args.version.strip(),
            }
        )

    passages = []
    for chunk in new_chunks:
        heading = chunk.get("headings", "")
        content = chunk.get("content", "")
        passages.append(f"{heading}\n{content}".strip())

    project_root = Path(__file__).resolve().parents[1]
    model_dir = Path(os.getenv("BGE_M3_DIR", "/hy-tmp/bge-m3"))
    if not model_dir.exists():
        model_dir = project_root / "model_assets" / "hf_models" / "BAAI--bge-m3"
    if not model_dir.exists():
        raise RuntimeError(f"local model not found: {model_dir}")
    print(
        json.dumps(
            {"stage": "load_model", "model": "BAAI/bge-m3", "path": str(model_dir)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    model = BGEM3FlagModel(str(model_dir), use_fp16=True)
    print(json.dumps({"stage": "encode", "passages": len(passages)}, ensure_ascii=False), flush=True)
    vectors = model.encode(passages, batch_size=args.batch_size, max_length=args.max_length)[
        "dense_vecs"
    ].astype("float32")

    if index_path.exists():
        index = faiss.read_index(str(index_path))
        if index.d != vectors.shape[1]:
            raise RuntimeError(
                f"embedding dim mismatch: index={index.d}, current={vectors.shape[1]}"
            )
    else:
        index = faiss.IndexFlatIP(vectors.shape[1])
    print(
        json.dumps({"stage": "index_ready", "ntotal_before": int(index.ntotal)}, ensure_ascii=False),
        flush=True,
    )

    vector_ids = np.array([int(item["chunk_id"]) for item in new_chunks], dtype="int64")
    try:
        index.add(vectors)
    except RuntimeError as exc:
        # Some FAISS index wrappers (e.g. IndexIDMap) require explicit IDs.
        if "add_with_ids" not in str(exc):
            raise
        if not hasattr(index, "add_with_ids"):
            raise
        index.add_with_ids(vectors, vector_ids)
    chunks.extend(new_chunks)

    # Keep chunk_ids.npy strictly aligned with chunks.json order.
    all_chunk_ids = []
    for i, item in enumerate(chunks):
        try:
            all_chunk_ids.append(int(item.get("chunk_id")))
        except Exception:
            all_chunk_ids.append(i)
    all_chunk_ids_np = np.array(all_chunk_ids, dtype="int64")

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(chunk_ids_path, all_chunk_ids_np)

    elapsed_ms = int((time.time() - t0) * 1000)

    print(
        json.dumps(
            {
                "ok": True,
                "added_chunks": len(new_chunks),
                "index_total": index.ntotal,
                "chunk_ids_total": int(all_chunk_ids_np.shape[0]),
                "source": args.source,
                "elapsed_ms": elapsed_ms,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
