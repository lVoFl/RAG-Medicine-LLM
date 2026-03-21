import { Router } from "express";
import jwt from "jsonwebtoken";
import {
  createMessage,
  ensureConversationAccessible,
  listMessagesByConversation,
  patchMessage,
} from "../services/messageService.js";

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

    await ensureConversationAccessible(conversationId, userId);
    const messages = await listMessagesByConversation(conversationId);
    res.json(messages);
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

    await ensureConversationAccessible(conversationId, userId);
    const createdMessage = await createMessage({ conversationId, role, content, tokens });
    res.status(201).json(createdMessage);
  } catch (err) {
    next(err);
  }
});

// PATCH /api/conversations/:id/messages/:messageId
// Typical usage: create message first, then patch tokenizer-based tokens from model usage.
router.patch("/:id/messages/:messageId", async (req, res, next) => {
  try {
    const userId = getUserIdFromAuthHeader(req);
    const conversationId = Number(req.params.id);
    const messageId = Number(req.params.messageId);
    const { tokens, content } = req.body || {};

    if (!Number.isInteger(conversationId)) {
      return res.status(400).json({ error: "invalid conversation id" });
    }
    if (!Number.isInteger(messageId)) {
      return res.status(400).json({ error: "invalid message id" });
    }

    if (tokens !== undefined && (!Number.isInteger(tokens) || tokens < 0)) {
      return res.status(400).json({ error: "tokens must be a non-negative integer" });
    }
    if (tokens === undefined && content === undefined) {
      return res.status(400).json({ error: "at least one of tokens or content is required" });
    }

    await ensureConversationAccessible(conversationId, userId);
    const updatedMessage = await patchMessage({ conversationId, messageId, content, tokens });
    res.json(updatedMessage);
  } catch (err) {
    next(err);
  }
});

export default router;
