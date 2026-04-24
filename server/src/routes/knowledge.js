import { Router } from "express";
import jwt from "jsonwebtoken";
import pool from "../db.js";
import path from "path";
import { spawn } from "child_process";
import fs from "fs/promises";
import crypto from "crypto";

const router = Router();
const JWT_SECRET = process.env.JWT_SECRET || "change-this-secret-in-production";
const MODEL_SERVICE_URL = process.env.MODEL_SERVICE_URL || "http://127.0.0.1:8001";
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
let isReindexRunning = false;

function getAuthUser(req) {
  const authHeader = req.headers.authorization || "";
  const [scheme, token] = authHeader.split(" ");
  if (scheme !== "Bearer" || !token) {
    const err = new Error("missing or invalid authorization header");
    err.status = 401;
    throw err;
  }

  try {
    return jwt.verify(token, JWT_SECRET);
  } catch {
    const err = new Error("invalid or expired token");
    err.status = 401;
    throw err;
  }
}

function ensureAdmin(req) {
  const user = getAuthUser(req);
  if (!user?.isAdmin) {
    const err = new Error("admin permission required");
    err.status = 403;
    throw err;
  }
  return user;
}

function normalizeTags(tags) {
  if (!Array.isArray(tags)) return [];
  return tags
    .map((tag) => String(tag || "").trim())
    .filter(Boolean)
    .slice(0, 20);
}

async function markKnowledgeIndexDirty() {
  await pool.query(
    `
    UPDATE knowledge_index_meta
    SET status = 'dirty', last_error = NULL
    WHERE id = 1
    `
  );
}

async function runCommand(command, args, cwd) {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    console.log(`[knowledge] run command: ${command} ${args.join(" ")} | cwd=${cwd}`);
    const child = spawn(command, args, { cwd, shell: false });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += String(chunk || "");
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk || "");
    });
    child.on("error", reject);
    child.on("close", (code) => {
      const elapsedMs = Date.now() - startedAt;
      if (code === 0) {
        console.log(
          `[knowledge] command success (${elapsedMs}ms): ${command} ${args[0] || ""}\n` +
            `[stdout]\n${stdout}\n[stderr]\n${stderr}`
        );
        resolve({ stdout, stderr, elapsedMs });
        return;
      }
      console.error(
        `[knowledge] command failed (${elapsedMs}ms): ${command} ${args.join(" ")}\n` +
          `[stdout]\n${stdout}\n[stderr]\n${stderr}`
      );
      reject(new Error(`${command} exited with code ${code}\n${stderr || stdout}`));
    });
  });
}

function buildIngestKeyFromSource(source) {
  return crypto.createHash("sha1").update(String(source || "").trim()).digest("hex");
}

async function reloadRagIndex() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  const startedAt = Date.now();
  console.log(`[knowledge] reload rag index via ${MODEL_SERVICE_URL}/rag/reload`);
  try {
    const reloadResponse = await fetch(`${MODEL_SERVICE_URL}/rag/reload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
      signal: controller.signal,
    });
    const rawBody = await reloadResponse.text();
    console.log(
      `[knowledge] reload rag response: status=${reloadResponse.status} elapsed=${Date.now() - startedAt}ms body=${rawBody}`
    );
    if (!reloadResponse.ok) {
      let data = {};
      try {
        data = JSON.parse(rawBody);
      } catch {
        // ignore
      }
      throw new Error(data.error || rawBody || "model service reload failed");
    }
  } catch (err) {
    if (err?.name === "AbortError") {
      throw new Error("model service reload timeout (15s)");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

// GET /api/knowledge/documents
router.get("/documents", async (req, res, next) => {
  try {
    ensureAdmin(req);
    const keyword = String(req.query.keyword || "").trim();
    const page = Math.max(Number(req.query.page) || 1, 1);
    const pageSize = Math.min(Math.max(Number(req.query.pageSize) || 10, 1), 50);
    const offset = (page - 1) * pageSize;

    const whereSql = keyword
      ? "WHERE is_deleted = FALSE AND (title ILIKE $1 OR category ILIKE $1 OR source ILIKE $1)"
      : "WHERE is_deleted = FALSE";
    const params = keyword ? [`%${keyword}%`] : [];

    const countResult = await pool.query(
      `
      SELECT COUNT(*)::INT AS total
      FROM medical_documents
      ${whereSql}
      `,
      params
    );

    const listResult = await pool.query(
      `
      SELECT
        id,
        title,
        category,
        source,
        version,
        tags,
        ingest_source,
        created_at,
        updated_at,
        updated_by
      FROM medical_documents
      ${whereSql}
      ORDER BY updated_at DESC, id DESC
      LIMIT ${pageSize}
      OFFSET ${offset}
      `,
      params
    );

    res.json({
      list: listResult.rows,
      pagination: {
        page,
        pageSize,
        total: countResult.rows[0]?.total || 0,
      },
    });
  } catch (err) {
    next(err);
  }
});

// GET /api/knowledge/documents/:id
router.get("/documents/:id", async (req, res, next) => {
  try {
    ensureAdmin(req);
    const id = Number(req.params.id);
    if (!Number.isInteger(id)) {
      return res.status(400).json({ error: "invalid document id" });
    }

    const result = await pool.query(
      `
      SELECT
        id,
        title,
        category,
        source,
        version,
        tags,
        ingest_source,
        created_at,
        updated_at,
        updated_by
      FROM medical_documents
      WHERE id = $1 AND is_deleted = FALSE
      `,
      [id]
    );

    const row = result.rows[0];
    if (!row) {
      return res.status(404).json({ error: "document not found" });
    }

    res.json(row);
  } catch (err) {
    next(err);
  }
});

// POST /api/knowledge/documents
router.post("/documents", async (req, res, next) => {
  try {
    const user = ensureAdmin(req);
    const { title, category, source, version, tags } = req.body || {};

    if (!title || !String(title).trim()) {
      return res.status(400).json({ error: "title is required" });
    }

    const result = await pool.query(
      `
      INSERT INTO medical_documents (title, category, source, version, tags, updated_by)
      VALUES ($1, $2, $3, $4, $5::jsonb, $6)
      RETURNING
        id,
        title,
        category,
        source,
        version,
        tags,
        ingest_source,
        created_at,
        updated_at,
        updated_by
      `,
      [
        String(title).trim(),
        category ? String(category).trim() : null,
        source ? String(source).trim() : null,
        version ? String(version).trim() : null,
        JSON.stringify(normalizeTags(tags)),
        user.id,
      ]
    );

    await markKnowledgeIndexDirty();
    res.status(201).json(result.rows[0]);
  } catch (err) {
    next(err);
  }
});

// POST /api/knowledge/upload-text
// Accepts plain text content, then chunk -> embed -> append to FAISS.
router.post("/upload-text", async (req, res, next) => {
  if (isReindexRunning) {
    return res.status(409).json({ error: "reindex/upload is already running" });
  }

  try {
    const user = ensureAdmin(req);
    isReindexRunning = true;
    await pool.query(
      `
      UPDATE knowledge_index_meta
      SET status = 'running', last_error = NULL
      WHERE id = 1
      `
    );

    const { title, source, text, category, version, tags } = req.body || {};
    const normalizedTitle = String(title || "").trim();
    const normalizedSource = String(source || "").trim();
    const normalizedText = String(text || "").trim();
    if (!normalizedTitle) {
      return res.status(400).json({ error: "title is required" });
    }
    if (!normalizedSource) {
      return res.status(400).json({ error: "source is required" });
    }
    if (!normalizedText) {
      return res.status(400).json({ error: "text is required" });
    }
    console.log(
      `[knowledge] upload-text request: title=${normalizedTitle} source=${normalizedSource} ` +
        `textLength=${normalizedText.length} category=${category || ""} version=${version || ""}`
    );

    const databaseDir = path.resolve(process.cwd(), "../database");
    const generatedDir = path.join(databaseDir, "generated");
    await fs.mkdir(generatedDir, { recursive: true });
    const tmpFileName = `upload_${Date.now()}.txt`;
    const tmpTextPath = path.join(generatedDir, tmpFileName);
    await fs.writeFile(tmpTextPath, normalizedText, "utf-8");

    try {
      await runCommand(
        PYTHON_BIN,
        [
          "append_text_to_faiss.py",
          "--text-file",
          tmpTextPath,
          "--source",
          normalizedSource,
          "--headings",
          normalizedTitle,
          "--category",
          category ? String(category).trim() : "",
          "--version",
          version ? String(version).trim() : "",
        ],
        databaseDir
      );
    } finally {
      await fs.rm(tmpTextPath, { force: true }).catch(() => {});
    }

    const ingestKey = buildIngestKeyFromSource(normalizedSource);
    const upsertResult = await pool.query(
      `
      INSERT INTO medical_documents (
        title, category, source, version, tags, updated_by,
        ingest_key, ingest_source, is_deleted
      )
      VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,'text_upload',FALSE)
      ON CONFLICT (ingest_key)
      DO UPDATE SET
        title = EXCLUDED.title,
        category = EXCLUDED.category,
        source = EXCLUDED.source,
        version = EXCLUDED.version,
        tags = EXCLUDED.tags,
        updated_by = EXCLUDED.updated_by,
        ingest_source = EXCLUDED.ingest_source,
        is_deleted = FALSE,
        updated_at = NOW()
      RETURNING
        id, title, category, source, version, tags,
        ingest_source, created_at, updated_at, updated_by
      `,
      [
        normalizedTitle,
        category ? String(category).trim() : null,
        normalizedSource,
        version ? String(version).trim() : null,
        JSON.stringify(normalizeTags(tags)),
        user.id,
        ingestKey,
      ]
    );

    await reloadRagIndex();
    await pool.query(
      `
      UPDATE knowledge_index_meta
      SET
        status = 'synced',
        last_reindexed_at = NOW(),
        last_error = NULL
      WHERE id = 1
      `
    );

    res.status(201).json({ ok: true, document: upsertResult.rows[0] });
  } catch (err) {
    await pool.query(
      `
      UPDATE knowledge_index_meta
      SET status = 'error', last_error = $1
      WHERE id = 1
      `,
      [String(err?.message || err || "unknown error")]
    );
    next(err);
  } finally {
    isReindexRunning = false;
  }
});

// PATCH /api/knowledge/documents/:id
router.patch("/documents/:id", async (req, res, next) => {
  try {
    const user = ensureAdmin(req);
    const id = Number(req.params.id);
    if (!Number.isInteger(id)) {
      return res.status(400).json({ error: "invalid document id" });
    }

    const { title, category, source, version, tags } = req.body || {};
    if (
      title === undefined &&
      category === undefined &&
      source === undefined &&
      version === undefined &&
      tags === undefined
    ) {
      return res.status(400).json({ error: "at least one field is required" });
    }

    const result = await pool.query(
      `
      UPDATE medical_documents
      SET
        title = COALESCE($1, title),
        category = COALESCE($2, category),
        source = COALESCE($3, source),
        version = COALESCE($4, version),
        tags = COALESCE($5::jsonb, tags),
        updated_by = $6
      WHERE id = $7 AND is_deleted = FALSE
      RETURNING
        id,
        title,
        category,
        source,
        version,
        tags,
        ingest_source,
        created_at,
        updated_at,
        updated_by
      `,
      [
        title === undefined ? null : String(title).trim(),
        category === undefined ? null : category ? String(category).trim() : null,
        source === undefined ? null : source ? String(source).trim() : null,
        version === undefined ? null : version ? String(version).trim() : null,
        tags === undefined ? null : JSON.stringify(normalizeTags(tags)),
        user.id,
        id,
      ]
    );

    const row = result.rows[0];
    if (!row) {
      return res.status(404).json({ error: "document not found" });
    }
    await markKnowledgeIndexDirty();
    res.json(row);
  } catch (err) {
    next(err);
  }
});

// DELETE /api/knowledge/documents/:id
router.delete("/documents/:id", async (req, res, next) => {
  try {
    const id = Number(req.params.id);
    ensureAdmin(req);
    if (!Number.isInteger(id)) {
      return res.status(400).json({ error: "invalid document id" });
    }

    const result = await pool.query(
      `
      UPDATE medical_documents
      SET is_deleted = TRUE
      WHERE id = $1 AND is_deleted = FALSE
      RETURNING id
      `,
      [id]
    );

    if (!result.rows[0]) {
      return res.status(404).json({ error: "document not found" });
    }
    await markKnowledgeIndexDirty();
    res.status(204).send();
  } catch (err) {
    next(err);
  }
});

// GET /api/knowledge/index-status
router.get("/index-status", async (req, res, next) => {
  try {
    ensureAdmin(req);
    const result = await pool.query(
      `
      SELECT id, status, last_reindexed_at, last_error, updated_at
      FROM knowledge_index_meta
      WHERE id = 1
      `
    );
    res.json(result.rows[0] || null);
  } catch (err) {
    next(err);
  }
});

// POST /api/knowledge/reindex
router.post("/reindex", async (req, res, next) => {
  if (isReindexRunning) {
    return res.status(409).json({ error: "reindex is already running" });
  }

  try {
    ensureAdmin(req);
    isReindexRunning = true;

    await pool.query(
      `
      UPDATE knowledge_index_meta
      SET status = 'running', last_error = NULL
      WHERE id = 1
      `
    );

    const databaseDir = path.resolve(process.cwd(), "../database");
    await runCommand(PYTHON_BIN, ["build_index.py"], databaseDir);

    await reloadRagIndex();

    await pool.query(
      `
      UPDATE knowledge_index_meta
      SET
        status = 'synced',
        last_reindexed_at = NOW(),
        last_error = NULL
      WHERE id = 1
      `
    );

    res.json({ ok: true });
  } catch (err) {
    await pool.query(
      `
      UPDATE knowledge_index_meta
      SET status = 'error', last_error = $1
      WHERE id = 1
      `,
      [String(err?.message || err || "unknown error")]
    );
    next(err);
  } finally {
    isReindexRunning = false;
  }
});

export default router;
