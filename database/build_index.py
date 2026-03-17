"""
Build a FAISS index from pre-computed embeddings and save it to disk.

Inputs  (from database/embeddings/):
  embeddings.npy  — float32 array of shape (N, 1024), L2-normalized
  chunks.json     — full chunk list aligned with embeddings

Outputs (to database/faiss_index/):
  index.faiss     — FAISS IndexFlatIP (inner-product = cosine on unit vectors)
  chunks.json     — copy of chunk list (kept alongside the index for portability)
"""

import json
import numpy as np
import faiss
from pathlib import Path

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
INDEX_DIR = Path(__file__).parent / "faiss_index"


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact inner-product index (cosine similarity for L2-normed vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    # Load embeddings
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    chunks_path = EMBEDDINGS_DIR / "chunks.json"

    print(f"Loading embeddings from {embeddings_path} ...")
    embeddings = np.load(embeddings_path).astype("float32")
    print(f"  shape: {embeddings.shape}")

    print(f"Loading chunks from {chunks_path} ...")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  total chunks: {len(chunks)}")

    assert len(chunks) == embeddings.shape[0], (
        f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
    )

    # Build index
    print("Building FAISS IndexFlatIP ...")
    index = build_faiss_index(embeddings)
    print(f"  index.ntotal = {index.ntotal}")

    # Save
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index → {index_path}")

    out_chunks_path = INDEX_DIR / "chunks.json"
    with open(out_chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks      → {out_chunks_path}")


if __name__ == "__main__":
    main()
