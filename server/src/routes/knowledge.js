import { Router } from "express";
import jwt from "jsonwebtoken";
import pool from "../db.js";

const router = Router();
const JWT_SECRET = process.env.JWT_SECRET || "change-this-secret-in-production";

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

// GET /api/knowledge/documents
router.get("/documents", async (req, res, next) => {
  try {
    ensureAdmin(req);
    const keyword = String(req.query.keyword || "").trim();
    const page = Math.max(Number(req.query.page) || 1, 1);
    const pageSize = Math.min(Math.max(Number(req.query.pageSize) || 10, 1), 50);
    const offset = (page - 1) * pageSize;

    const whereSql = keyword
      ? "WHERE is_deleted = FALSE AND (title ILIKE $1 OR category ILIKE $1 OR summary ILIKE $1)"
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
        summary,
        content,
        source,
        version,
        tags,
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
        summary,
        content,
        source,
        version,
        tags,
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
    const { title, category, summary, content, source, version, tags } = req.body || {};

    if (!title || !String(title).trim()) {
      return res.status(400).json({ error: "title is required" });
    }
    if (!content || !String(content).trim()) {
      return res.status(400).json({ error: "content is required" });
    }

    const result = await pool.query(
      `
      INSERT INTO medical_documents (title, category, summary, content, source, version, tags, updated_by)
      VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
      RETURNING
        id,
        title,
        category,
        summary,
        content,
        source,
        version,
        tags,
        created_at,
        updated_at,
        updated_by
      `,
      [
        String(title).trim(),
        category ? String(category).trim() : null,
        summary ? String(summary).trim() : null,
        String(content),
        source ? String(source).trim() : null,
        version ? String(version).trim() : null,
        JSON.stringify(normalizeTags(tags)),
        user.id,
      ]
    );

    res.status(201).json(result.rows[0]);
  } catch (err) {
    next(err);
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

    const { title, category, summary, content, source, version, tags } = req.body || {};
    if (
      title === undefined &&
      category === undefined &&
      summary === undefined &&
      content === undefined &&
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
        summary = COALESCE($3, summary),
        content = COALESCE($4, content),
        source = COALESCE($5, source),
        version = COALESCE($6, version),
        tags = COALESCE($7::jsonb, tags),
        updated_by = $8
      WHERE id = $9 AND is_deleted = FALSE
      RETURNING
        id,
        title,
        category,
        summary,
        content,
        source,
        version,
        tags,
        created_at,
        updated_at,
        updated_by
      `,
      [
        title === undefined ? null : String(title).trim(),
        category === undefined ? null : category ? String(category).trim() : null,
        summary === undefined ? null : summary ? String(summary).trim() : null,
        content === undefined ? null : String(content),
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
    res.status(204).send();
  } catch (err) {
    next(err);
  }
});

export default router;
