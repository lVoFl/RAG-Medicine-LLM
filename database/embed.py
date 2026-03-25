"""
Embedding generator using BAAI/bge-m3 for FAISS vector database storage.
Reads all processed JSON chunks, encodes them with BGE-M3, and saves:
  - embeddings.npy  : float32 array of shape (N, 1024)
  - metadata.json   : list of chunk metadata dicts (without content)
  - chunks.json     : full chunks including content, aligned with embeddings
"""

import json
import time
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel

PROCESSED_DIRS = [
    Path(__file__).parent / "processed_pdf",
    Path(__file__).parent / "processed_txt",
]
OUTPUT_DIR = Path(__file__).parent / "embeddings"
BATCH_SIZE = 12
MAX_LENGTH = 8192


def load_model() -> BGEM3FlagModel:
    print("Loading BAAI/bge-m3 ...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("Model loaded.")
    return model


def load_all_chunks(processed_dir: Path) -> list[dict]:
    chunks = []
    for path in sorted(processed_dir.rglob("*.json")):
        if path.name == "index.json":
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                print(f"  Loaded {len(data):>4} chunks from {path.relative_to(processed_dir)}")
                chunks.extend(data)
            else:
                print(f"  [skip] {path.name}")
        except Exception as e:
            print(f"  [error] {path}: {e}")
    return chunks


def build_passage_text(chunk: dict) -> str:
    heading = chunk.get("headings", "").strip()
    content = chunk.get("content", "").strip()
    return f"{heading}\n{content}" if heading else content


def main():
    t0 = time.time()

    print("=" * 60)
    print("Step 1: Loading chunks ...")
    chunks = []
    for processed_dir in PROCESSED_DIRS:
        print(f"  Reading from: {processed_dir}")
        chunks.extend(load_all_chunks(processed_dir))
    if not chunks:
        print("No chunks found. Check PROCESSED_DIRS paths.")
        return
    print(f"  Total chunks: {len(chunks)}")

    print("\nStep 2: Building passage texts ...")
    passages = [build_passage_text(c) for c in chunks]

    print("\nStep 3: Loading model ...")
    model = load_model()

    print(f"\nStep 4: Encoding {len(passages)} passages (batch_size={BATCH_SIZE}) ...")
    embeddings = model.encode(
        passages,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )["dense_vecs"].astype("float32")
    print(f"  Embeddings shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    print("\nStep 5: Saving outputs ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_path = OUTPUT_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"  Saved embeddings  -> {emb_path}  ({emb_path.stat().st_size / 1e6:.1f} MB)")

    chunks_path = OUTPUT_DIR / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  Saved chunks      -> {chunks_path}")

    metadata = [{k: v for k, v in c.items() if k != "content"} for c in chunks]
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Saved metadata    -> {meta_path}")

    print(f"\nDone in {time.time() - t0:.1f}s.")
    print("=" * 60)


if __name__ == "__main__":
    main()
