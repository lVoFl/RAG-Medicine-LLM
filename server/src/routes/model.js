import { Router } from "express";
import jwt from "jsonwebtoken";
import { generateWithLocalModel } from "../services/modelClient.js";

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

    res.json({
      answer: result.answer,
      usage: result.usage || null,
      params: result.params || null,
    });
  } catch (err) {
    next(err);
  }
});

export default router;
