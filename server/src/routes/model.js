import { Router } from "express";
import jwt from "jsonwebtoken";
import { generateWithLocalModel } from "../services/modelClient.js";
import { runConversationWorkflow } from "../services/chatWorkflowService.js";
import pool from "../db.js"

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

// POST /api/model/generate
router.post("/generate", async (req, res, next) => {
  try {
    getUserIdFromAuthHeader(req);

    const {
      question,
      context,
      system_prompt: systemPrompt,
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: topP,
    } = req.body || {};

    if (!question || !String(question).trim()) {
      return res.status(400).json({ error: "question is required" });
    }

    const result = await generateWithLocalModel({
      question: String(question),
      context: context == null ? undefined : String(context),
      systemPrompt: systemPrompt == null ? undefined : String(systemPrompt),
      maxNewTokens,
      temperature,
      topP,
    });
    console.log(result)
    const usage =
      result.usage && typeof result.usage === "object"
        ? {
            ...result.usage,
            total_tokens:
              Number(result.usage.prompt_tokens || 0) + Number(result.usage.completion_tokens || 0),
          }
        : null;

    res.json({
      answer: result.answer,
      usage,
      params: result.params || null,
    });
  } catch (err) {
    next(err);
  }
});

// POST /api/model/conversations/:id/generate
// Workflow: check conversation -> insert user message -> call local LLM -> insert assistant message -> return response.
router.post("/conversations/:id/generate", async (req, res, next) => {
  try {
    const userId = getUserIdFromAuthHeader(req);
    const conversationId = Number(req.params.id);
    if (!Number.isInteger(conversationId)) {
      return res.status(400).json({ error: "invalid conversation id" });
    }

    const {
      question,
      context,
      system_prompt: systemPrompt,
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: topP,
    } = req.body || {};

    const normalizedQuestion = String(question || "").trim();
    if (!normalizedQuestion) {
      return res.status(400).json({ error: "question is required" });
    }

    const result = await runConversationWorkflow({
      userId,
      conversationId,
      question: normalizedQuestion,
      context: context == null ? undefined : String(context),
      systemPrompt: systemPrompt == null ? undefined : String(systemPrompt),
      maxNewTokens,
      temperature,
      topP,
    });
    console.log(result);
    const updateResult = await pool.query(
      `
      UPDATE conversations
      SET
        last_message = COALESCE($1::jsonb, last_message)
      WHERE id = $2 AND user_id = $3 AND is_deleted = FALSE
      RETURNING id
      `,
      [result.assistant_message.content, conversationId, userId]
    );
    res.status(201).json(result);
    
  } catch (err) {
    next(err);
  }
});

export default router;
