import { Router } from "express";
import jwt from "jsonwebtoken";
import pool from "../db.js";

const router = Router();

const JWT_SECRET = process.env.JWT_SECRET || "change-this-secret-in-production";

const conversationSelectSql = `
  SELECT
    c.id,
    c.user_id,
    c.title,
    COALESCE(lm.content, c.last_message) AS last_message,
    COALESCE(mc.cnt, 0)::INT AS message_count,
    c.created_at,
    c.updated_at
  FROM conversations c
  LEFT JOIN LATERAL (
    SELECT m.content
    FROM messages m
    WHERE m.conversation_id = c.id
    ORDER BY m.created_at DESC, m.id DESC
    LIMIT 1
  ) lm ON TRUE
  LEFT JOIN LATERAL (
    SELECT COUNT(*)::INT AS cnt
    FROM messages m2
    WHERE m2.conversation_id = c.id
  ) mc ON TRUE
`;

function getUserIdFromAuthHeader(req) {
  const authHeader = req.headers.authorization || "";
  const [scheme, token] = authHeader.split(" ");
  if (scheme !== "Bearer" || !token) {
    const err = new Error("missing or invalid authorization header");
    err.status = 401;
    throw err;
  }

  try {
    const payload = jwt.verify(token, JWT_SECRET);
    return payload.id;
  } catch {
    const err = new Error("invalid or expired token");
    err.status = 401;
    throw err;
  }
}

// GET /api/conversations
router.get("/", async (req, res, next) => {
  try {
    const userId = getUserIdFromAuthHeader(req);
    const result = await pool.query(
      `
      ${conversationSelectSql}
      WHERE c.user_id = $1 AND c.is_deleted = FALSE
      ORDER BY c.updated_at DESC
      `,
      [userId]
    );
    res.json(result.rows);
  } catch (err) {
    next(err);
  }
});

// POST /api/conversations
router.post("/", async (req, res, next) => {
  try {
    const userId = getUserIdFromAuthHeader(req);
    const { title } = req.body;

    const result = await pool.query(
      `
      INSERT INTO conversations (user_id, title)
      VALUES ($1, $2)
      RETURNING id, user_id, title, created_at, updated_at
      `,
      [userId, title ?? null]
    );
    const created = result.rows[0];
    res.status(201).json({
      ...created,
      last_message: null,
      message_count: 0,
    });
  } catch (err) {
    next(err);
  }
});

// GET /api/conversations/:id
router.get("/:id", async (req, res, next) => {
  try {
    const userId = getUserIdFromAuthHeader(req);
    const conversationId = Number(req.params.id);
    if (!Number.isInteger(conversationId)) {
      return res.status(400).json({ error: "invalid conversation id" });
    }

    const result = await pool.query(
      `
      ${conversationSelectSql}
      WHERE c.id = $1 AND c.user_id = $2 AND c.is_deleted = FALSE
      `,
      [conversationId, userId]
    );

    const conversation = result.rows[0];
    if (!conversation) {
      return res.status(404).json({ error: "conversation not found" });
    }

    res.json(conversation);
  } catch (err) {
    next(err);
  }
});

// PATCH /api/conversations/:id
router.patch("/:id", async (req, res, next) => {
  try {
    const userId = getUserIdFromAuthHeader(req);
    const conversationId = Number(req.params.id);
    const { title, last_message } = req.body;
    const normalizedLastMessage =
      last_message === undefined || last_message === null ? null : JSON.stringify(last_message);

    if (!Number.isInteger(conversationId)) {
      return res.status(400).json({ error: "invalid conversation id" });
    }

    const updateResult = await pool.query(
      `
      UPDATE conversations
      SET
        title = COALESCE($1, title),
        last_message = COALESCE($2::jsonb, last_message)
      WHERE id = $3 AND user_id = $4 AND is_deleted = FALSE
      RETURNING id
      `,
      [title ?? null, normalizedLastMessage, conversationId, userId]
    );

    if (!updateResult.rows[0]) {
      return res.status(404).json({ error: "conversation not found" });
    }

    const result = await pool.query(
      `
      ${conversationSelectSql}
      WHERE c.id = $1 AND c.user_id = $2 AND c.is_deleted = FALSE
      `,
      [conversationId, userId]
    );
    const conversation = result.rows[0];
    res.json(conversation);
  } catch (err) {
    next(err);
  }
});

// DELETE /api/conversations/:id (soft delete)
router.delete("/:id", async (req, res, next) => {
  try {
    const userId = getUserIdFromAuthHeader(req);
    const conversationId = Number(req.params.id);
    if (!Number.isInteger(conversationId)) {
      return res.status(400).json({ error: "invalid conversation id" });
    }

    const result = await pool.query(
      `
      UPDATE conversations
      SET is_deleted = TRUE
      WHERE id = $1 AND user_id = $2 AND is_deleted = FALSE
      RETURNING id
      `,
      [conversationId, userId]
    );

    if (!result.rows[0]) {
      return res.status(404).json({ error: "conversation not found" });
    }

    res.status(204).send();
  } catch (err) {
    next(err);
  }
});

export default router;
