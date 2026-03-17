import { Router } from "express";
import jwt from "jsonwebtoken";
import pool from "../db.js";

const router = Router();

const JWT_SECRET = process.env.JWT_SECRET || "change-this-secret-in-production";

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

// GET /api/conversations/:id/messages
router.get("/:id/messages", async (req, res, next) => {
  try {
    const userId = getUserIdFromAuthHeader(req);
    const conversationId = Number(req.params.id);
    if (!Number.isInteger(conversationId)) {
      return res.status(400).json({ error: "invalid conversation id" });
    }

    const conversationResult = await pool.query(
      `
      SELECT id
      FROM conversations
      WHERE id = $1 AND user_id = $2 AND is_deleted = FALSE
      `,
      [conversationId, userId]
    );
    if (!conversationResult.rows[0]) {
      return res.status(404).json({ error: "conversation not found" });
    }

    const result = await pool.query(
      `
      SELECT id, conversation_id, role, content, tokens, created_at
      FROM messages
      WHERE conversation_id = $1
      ORDER BY created_at ASC, id ASC
      `,
      [conversationId]
    );

    res.json(result.rows);
  } catch (err) {
    next(err);
  }
});

// POST /api/conversations/:id/messages
router.post("/:id/messages", async (req, res, next) => {
  try {
    const userId = getUserIdFromAuthHeader(req);
    const conversationId = Number(req.params.id);
    const { role, content, tokens } = req.body;

    if (!Number.isInteger(conversationId)) {
      return res.status(400).json({ error: "invalid conversation id" });
    }
    if (!role || content === undefined || content === null) {
      return res.status(400).json({ error: "role and content are required" });
    }

    const conversationResult = await pool.query(
      `
      SELECT id
      FROM conversations
      WHERE id = $1 AND user_id = $2 AND is_deleted = FALSE
      `,
      [conversationId, userId]
    );
    if (!conversationResult.rows[0]) {
      return res.status(404).json({ error: "conversation not found" });
    }

    const insertResult = await pool.query(
      `
      INSERT INTO messages (conversation_id, role, content, tokens)
      VALUES ($1, $2, $3, $4)
      RETURNING id, conversation_id, role, content, tokens, created_at
      `,
      [conversationId, role, content, tokens ?? null]
    );

    res.status(201).json(insertResult.rows[0]);
  } catch (err) {
    next(err);
  }
});

export default router;
