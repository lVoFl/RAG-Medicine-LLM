"""
Delete chunks by `source` from FAISS index files.

Files updated:
  - index.faiss
  - chunks.json
  - chunk_ids.npy
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Delete one source from FAISS index.")
    parser.add_argument("--source", type=str, required=True, help="source to delete")
    parser.add_argument("--index-dir", type=str, default=str(Path(__file__).parent / "faiss_index"))
    parser.add_argument("--backup", action="store_true", help="write backup files before overwrite")
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source.strip()
    if not source:
        raise ValueError("source is empty")

    index_dir = Path(args.index_dir)
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.json"
    chunk_ids_path = index_dir / "chunk_ids.npy"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("index.faiss or chunks.json not found")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise RuntimeError("chunks.json format invalid")

    total_before = len(chunks)
    keep_pos = [i for i, c in enumerate(chunks) if str(c.get("source", "")).strip() != source]
    del_pos = [i for i, c in enumerate(chunks) if str(c.get("source", "")).strip() == source]
    deleted = len(del_pos)
    if deleted == 0:
        print(
            json.dumps(
                {"ok": True, "deleted": 0, "message": f"source not found: {source}"},
                ensure_ascii=False,
            )
        )
        return

    # Rebuild index from kept vectors via reconstruct() to avoid re-embedding.
    dim = index.d
    kept_vecs = np.zeros((len(keep_pos), dim), dtype="float32")
    for out_i, old_i in enumerate(keep_pos):
        vec = index.reconstruct(int(old_i))
        kept_vecs[out_i] = np.asarray(vec, dtype="float32")

    new_index = faiss.IndexFlatIP(dim)
    if len(keep_pos) > 0:
        new_index.add(kept_vecs)

    new_chunks = [chunks[i] for i in keep_pos]
    for i, item in enumerate(new_chunks):
        item["faiss_id"] = i
    new_chunk_ids = np.arange(len(new_chunks), dtype="int64")

    if args.backup:
        backup_dir = index_dir / "backup_delete_source"
        backup_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(backup_dir / "index.faiss.bak"))
        with open(backup_dir / "chunks.json.bak", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        if chunk_ids_path.exists():
            old_ids = np.load(chunk_ids_path)
            np.save(backup_dir / "chunk_ids.npy.bak", old_ids)

    faiss.write_index(new_index, str(index_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(new_chunks, f, ensure_ascii=False, indent=2)
    np.save(chunk_ids_path, new_chunk_ids)

    print(
        json.dumps(
            {
                "ok": True,
                "source": source,
                "deleted": deleted,
                "total_before": total_before,
                "total_after": len(new_chunks),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

