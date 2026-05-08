"""
Build a FAISS index from pre-computed embeddings and save it to disk.

Inputs  (from database/embeddings/):
  embeddings.npy  - float32 array of shape (N, D), L2-normalized
  chunks.json     - full chunk list aligned with embeddings

Outputs (to database/faiss_index/):
  index.faiss     - FAISS IndexIDMap2(IndexFlatIP), supports add/remove by id
  chunks.json     - copy of chunk list (kept alongside the index for portability)
  chunk_ids.npy   - int64 ids aligned with chunks (position -> faiss id)

Default behavior:
  After writing FAISS files, sync chunks back to PostgreSQL directly.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
INDEX_DIR = Path(__file__).parent / "faiss_index"


def parse_args():
    """Parse CLI args for index build and optional DB sync."""
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings.")
    parser.add_argument("--embeddings-dir", type=str, default=str(EMBEDDINGS_DIR))
    parser.add_argument("--index-dir", type=str, default=str(INDEX_DIR))
    parser.add_argument(
        "--skip-db-sync",
        action="store_true",
        help="Skip syncing FAISS chunks back to PostgreSQL.",
    )
    parser.add_argument(
        "--server-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "server"),
        help="Server directory that contains .env",
    )
    return parser.parse_args()


def build_faiss_index(embeddings: np.ndarray, chunk_ids: np.ndarray) -> faiss.Index:
    """Build an ID-aware IP index from unit-normalized embeddings.

    Using IndexIDMap2 allows future incremental add/remove by explicit ids:
    - add_with_ids(vectors, ids)
    - remove_ids(IDSelectorBatch(ids))
    """
    dim = embeddings.shape[1]
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(embeddings, chunk_ids.astype("int64"))
    return index


def _build_chunk_ids(chunks: list[dict]) -> np.ndarray:
    """Resolve FAISS internal ids strictly from `faiss_id`.

    Rules:
    1) FAISS internal id is always `faiss_id`
    2) `chunk_id` is metadata only (must never be used as internal id)
    3) If `faiss_id` missing, assign by current position for determinism
    """
    ids = np.empty(len(chunks), dtype="int64")
    used_ids: set[int] = set()

    for i, chunk in enumerate(chunks):
        raw_id = chunk.get("faiss_id")
        if raw_id is None:
            faiss_id = i
        else:
            try:
                faiss_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid faiss_id at chunk[{i}]: {raw_id!r}") from exc
            if faiss_id < 0:
                raise ValueError(f"faiss_id must be >= 0 at chunk[{i}], got {faiss_id}")

        if faiss_id in used_ids:
            raise ValueError(f"Duplicate faiss_id detected: {faiss_id}")
        used_ids.add(faiss_id)
        ids[i] = faiss_id
        # Persist canonical id in chunks.json for portability and easier debugging.
        chunk["faiss_id"] = int(faiss_id)

    return ids


def _validate_chunk_schema(chunks: list[dict]) -> None:
    """Fail fast when chunks contain mixed/non-canonical structures."""
    required = {"chunk_id", "source", "category", "headings", "content"}
    bad_examples: list[str] = []
    bad_count = 0

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            bad_count += 1
            if len(bad_examples) < 3:
                bad_examples.append(f"[{i}] non-dict item: {type(chunk).__name__}")
            continue
        missing = [k for k in required if k not in chunk]
        if missing:
            bad_count += 1
            if len(bad_examples) < 3:
                bad_examples.append(f"[{i}] missing={missing}, keys={sorted(chunk.keys())}")

    if bad_count:
        raise ValueError(
            "Invalid chunk schema detected in embeddings/chunks.json. "
            f"bad_items={bad_count}/{len(chunks)}. Examples: " + " | ".join(bad_examples)
        )


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Load a simple KEY=VALUE style .env file into a dict."""
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _load_db_config(server_dir: Path) -> dict[str, Any]:
    """Resolve DB connection config from server/.env.

    Priority:
    1) DATABASE_URL (returned as dsn)
    2) DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME
    """
    env_data = _parse_env_file(server_dir / ".env")
    database_url = env_data.get("DATABASE_URL")
    if database_url:
        return {"dsn": database_url}
    return {
        "host": env_data.get("DB_HOST", "localhost"),
        "port": int(env_data.get("DB_PORT", "5432")),
        "user": env_data.get("DB_USER", "postgres"),
        "password": env_data.get("DB_PASSWORD", ""),
        "dbname": env_data.get("DB_NAME", "app"),
    }


def _connect_pg(db_config: dict[str, Any]):
    """Connect to PostgreSQL using psycopg2 first, then psycopg fallback."""
    try:
        import psycopg2  # type: ignore

        return psycopg2.connect(**db_config)
    except ModuleNotFoundError:
        try:
            import psycopg  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DB sync requires psycopg2 or psycopg. Install one in current Python env."
            ) from exc
        return psycopg.connect(**db_config)


def _normalize(value: Any) -> str:
    """Convert any value to stripped string; None -> empty string."""
    return str(value or "").strip()


def _chunk_source(chunk: dict) -> str:
    """Read source from chunk JSON as the canonical DB source field."""
    return _normalize(chunk.get("source")) or "未知来源"


def _chunk_title(chunk: dict) -> str:
    return chunk.get("title") or "未知"


def _ingest_key(chunk: dict) -> str:
    # DB only keeps one record per source document, so source is the dedup basis.
    source = _chunk_source(chunk)
    base = source
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _aggregate_chunks(chunks: list[dict]) -> list[dict]:
    """Aggregate chunk-level rows into source-level document rows for DB sync."""
    grouped: dict[str, dict] = {}
    for chunk in chunks:
        key = _ingest_key(chunk)
        category = _normalize(chunk.get("category"))
        if key not in grouped:
            source = _chunk_source(chunk)
            title = _chunk_title(chunk)
            grouped[key] = {
                "ingest_key": key,
                "title": title,
                "category": category or None,
                "source": source,
                "version": None,
                "tag_set": set(),
            }
        if category:
            grouped[key]["tag_set"].add(category)

    docs: list[dict] = []
    for item in grouped.values():
        tags = sorted(item["tag_set"]) if item["tag_set"] else []
        docs.append(
            {
                "ingest_key": item["ingest_key"],
                "title": item["title"],
                "category": item["category"],
                "source": item["source"],
                "version": item["version"],
                "tags": tags,
            }
        )
    return docs


def _sync_chunks_to_db(chunks: list[dict], server_dir: Path):
    """Upsert aggregated docs into PostgreSQL and mark index sync status."""
    docs = _aggregate_chunks(chunks)
    db_config = _load_db_config(server_dir)
    conn = _connect_pg(db_config)

    try:
        with conn:
            with conn.cursor() as cur:
                # Ensure sync-tracking columns exist on target table.
                cur.execute(
                    """
                    ALTER TABLE medical_documents
                    ADD COLUMN IF NOT EXISTS ingest_key VARCHAR(64)
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE medical_documents
                    ADD COLUMN IF NOT EXISTS ingest_source VARCHAR(32) NOT NULL DEFAULT 'manual'
                    """
                )
                cur.execute(
                    # If an old partial unique index exists, drop it first.
                    """
                    DO $$
                    BEGIN
                      IF EXISTS (
                        SELECT 1
                        FROM pg_indexes
                        WHERE schemaname = 'public'
                          AND indexname = 'idx_medical_documents_ingest_key'
                          AND indexdef ILIKE '%WHERE ingest_key IS NOT NULL%'
                      ) THEN
                        DROP INDEX idx_medical_documents_ingest_key;
                      END IF;
                    END $$;
                    """
                )
                cur.execute(
                    # Unique ingest_key enforces one row per source document.
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_medical_documents_ingest_key
                    ON medical_documents(ingest_key)
                    """
                )
                cur.execute(
                    # Metadata table tracks whether FAISS sync is fresh or errored.
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_index_meta (
                      id SMALLINT PRIMARY KEY DEFAULT 1,
                      status VARCHAR(20) NOT NULL DEFAULT 'dirty',
                      last_reindexed_at TIMESTAMPTZ,
                      last_error TEXT,
                      updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO knowledge_index_meta (id, status)
                    VALUES (1, 'dirty')
                    ON CONFLICT (id) DO NOTHING
                    """
                )

                # Full refresh mode: clear previous FAISS-synced rows before re-insert.
                cur.execute(
                    """
                    DELETE FROM medical_documents
                    WHERE ingest_source = 'faiss_sync'
                    """
                )

                for item in docs:
                    # Upsert each aggregated doc row by ingest_key.
                    cur.execute(
                        """
                        INSERT INTO medical_documents (
                          title, category, source, version, tags,
                          ingest_key, ingest_source, is_deleted
                        )
                        VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,FALSE)
                        ON CONFLICT (ingest_key)
                        DO UPDATE SET
                          title = EXCLUDED.title,
                          category = EXCLUDED.category,
                          source = EXCLUDED.source,
                          version = EXCLUDED.version,
                          tags = EXCLUDED.tags,
                          ingest_source = EXCLUDED.ingest_source,
                          is_deleted = FALSE,
                          updated_at = NOW()
                        """,
                        (
                            item["title"],
                            item["category"],
                            item["source"],
                            item["version"],
                            json.dumps(item["tags"], ensure_ascii=False),
                            item["ingest_key"],
                            "faiss_sync",
                        ),
                    )

                cur.execute(
                    # Mark sync metadata as healthy after successful transaction.
                    """
                    UPDATE knowledge_index_meta
                    SET
                      status = 'synced',
                      last_reindexed_at = NOW(),
                      last_error = NULL
                    WHERE id = 1
                    """
                )
    except Exception as exc:
        # Best-effort error status update; do not mask original exception.
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE knowledge_index_meta
                        SET status = 'error', last_error = %s
                        WHERE id = 1
                        """,
                        (str(exc),),
                    )
        except Exception:
            pass
        raise
    finally:
        conn.close()

    print(f"Database sync completed. synced docs: {len(docs)}")


def main():
    """Entry point: load files, build index, save artifacts, optional DB sync."""
    args = parse_args()
    embeddings_dir = Path(args.embeddings_dir)
    index_dir = Path(args.index_dir)

    embeddings_path = embeddings_dir / "embeddings.npy"
    chunks_path = embeddings_dir / "chunks.json"

    print(f"Loading embeddings from {embeddings_path} ...")
    embeddings = np.load(embeddings_path).astype("float32")
    print(f"  shape: {embeddings.shape}")

    print(f"Loading chunks from {chunks_path} ...")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  total chunks: {len(chunks)}")
    _validate_chunk_schema(chunks)

    assert len(chunks) == embeddings.shape[0], (
        # chunks.json and embeddings.npy are expected to be positionally aligned.
        f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
    )

    chunk_ids = _build_chunk_ids(chunks)

    print("Building FAISS IndexIDMap2(IndexFlatIP) ...")
    index = build_faiss_index(embeddings, chunk_ids)
    print(f"  index.ntotal = {index.ntotal}")

    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index -> {index_path}")

    out_chunks_path = index_dir / "chunks.json"
    with open(out_chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks      -> {out_chunks_path}")

    chunk_ids_path = index_dir / "chunk_ids.npy"
    np.save(chunk_ids_path, chunk_ids.astype("int64"))
    print(f"Saved chunk ids   -> {chunk_ids_path}")

    if not args.skip_db_sync:
        print("Syncing chunks to PostgreSQL directly ...")
        _sync_chunks_to_db(chunks, Path(args.server_dir).resolve())


if __name__ == "__main__":
    main()
