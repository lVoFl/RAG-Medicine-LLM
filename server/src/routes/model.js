import { Router } from "express";
import jwt from "jsonwebtoken";
import { generateWithLocalModel, generateWithLocalModelStream } from "../services/modelClient.js";
import {
  persistConversationGeneration,
  prepareConversationGeneration,
  runConversationWorkflow,
} from "../services/chatWorkflowService.js";
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
      history,
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
      history: Array.isArray(history) ? history : undefined,
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
    await pool.query(
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

router.post("/conversations/:id/generate/stream", async (req, res, next) => {
  let streamSession = null;
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

    const { history, userMessage, baseContext } = await prepareConversationGeneration({
      userId,
      conversationId,
      question: normalizedQuestion,
      context: context == null ? undefined : String(context),
      systemPrompt: systemPrompt == null ? undefined : String(systemPrompt),
      maxNewTokens,
    });

    streamSession = await generateWithLocalModelStream({
      question: normalizedQuestion,
      context: baseContext || undefined,
      history: history.length > 0 ? history : undefined,
      systemPrompt: systemPrompt == null ? undefined : String(systemPrompt),
      maxNewTokens,
      temperature,
      topP,
    });

    res.status(200);
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders?.();

    const sendEvent = (payload) => {
      res.write(`data: ${JSON.stringify(payload)}\n\n`);
    };

    const decoder = new TextDecoder();
    const reader = streamSession.response.body.getReader();
    let buffer = "";
    let finalAnswer = "";
    let finalUsage = null;
    let finalParams = null;
    let finalRetrievedDocs = [];

    const abortUpstream = () => {
      try {
        streamSession?.abort?.();
      } catch {
        // ignore
      }
    };
    req.on("close", abortUpstream);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const dataLine = frame
          .split("\n")
          .map((line) => line.trim())
          .find((line) => line.startsWith("data:"));

        if (!dataLine) {
          boundary = buffer.indexOf("\n\n");
          continue;
        }

        const rawData = dataLine.slice(5).trim();
        if (!rawData) {
          boundary = buffer.indexOf("\n\n");
          continue;
        }

        let event;
        try {
          event = JSON.parse(rawData);
        } catch {
          boundary = buffer.indexOf("\n\n");
          continue;
        }

        if (event?.type === "delta" && typeof event.text === "string") {
          finalAnswer += event.text;
        } else if (event?.type === "end") {
          finalAnswer = String(event.answer || finalAnswer || "");
          finalUsage = event.usage || null;
          finalParams = event.params || null;
          finalRetrievedDocs = Array.isArray(event.retrieved_docs) ? event.retrieved_docs : [];
        } else if (event?.type === "error") {
          sendEvent(event);
          res.end();
          return;
        }

        sendEvent(event);
        boundary = buffer.indexOf("\n\n");
      }
    }

    const persisted = await persistConversationGeneration({
      conversationId,
      userMessage,
      answer: finalAnswer,
      usage: finalUsage,
      params: finalParams,
      retrievedDocs: finalRetrievedDocs,
    });

    await pool.query(
      `
      UPDATE conversations
      SET
        last_message = COALESCE($1::jsonb, last_message)
      WHERE id = $2 AND user_id = $3 AND is_deleted = FALSE
      RETURNING id
      `,
      [persisted.assistant_message.content, conversationId, userId]
    );

    sendEvent({
      type: "persisted",
      user_message: persisted.user_message,
      assistant_message: persisted.assistant_message,
      usage: persisted.usage,
      params: persisted.params,
    });
    res.end();
  } catch (err) {
    if (res.headersSent) {
      res.write(`data: ${JSON.stringify({ type: "error", error: err.message || "stream failed" })}\n\n`);
      res.end();
    } else {
      next(err);
    }
  } finally {
    streamSession?.clearTimeout?.();
  }
});

export default router;
