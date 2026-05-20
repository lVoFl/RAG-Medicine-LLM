import { Router } from "express";
import jwt from "jsonwebtoken";
import multer from "multer";
import fs from "fs/promises";
import os from "os";
import path from "path";
import { spawn } from "child_process";
import readline from "readline";

const router = Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
});

const JWT_SECRET = process.env.JWT_SECRET || "change-this-secret-in-production";
const ASR_LOCAL_SCRIPT = process.env.ASR_LOCAL_SCRIPT || "scripts/asr_local.py";
const ASR_PYTHON_BIN = process.env.ASR_PYTHON_BIN || "python";
const ASR_LOCAL_MODEL_SIZE = process.env.ASR_LOCAL_MODEL_SIZE || "small";
const ASR_LOCAL_DEVICE = process.env.ASR_LOCAL_DEVICE || "auto";
const ASR_LOCAL_COMPUTE_TYPE = process.env.ASR_LOCAL_COMPUTE_TYPE || "auto";
const ASR_LOCAL_LANGUAGE = process.env.ASR_LOCAL_LANGUAGE || "zh";
const ASR_LOCAL_TIMEOUT_MS = Number(process.env.ASR_LOCAL_TIMEOUT_MS || 180000);

let asrWorker = null;
let workerRl = null;
let workerStartPromise = null;
const pendingRequests = new Map();

function startAsrWorker() {
  if (asrWorker && !asrWorker.killed) return Promise.resolve();
  if (workerStartPromise) return workerStartPromise;

  workerStartPromise = new Promise((resolve, reject) => {
    const args = [
      ASR_LOCAL_SCRIPT,
      "--serve",
      "--model",
      ASR_LOCAL_MODEL_SIZE,
      "--device",
      ASR_LOCAL_DEVICE,
      "--compute-type",
      ASR_LOCAL_COMPUTE_TYPE,
      "--language",
      ASR_LOCAL_LANGUAGE,
    ];

    const child = spawn(ASR_PYTHON_BIN, args, {
      cwd: path.resolve(process.cwd()),
      windowsHide: true,
    });
    asrWorker = child;

    const stderrLines = [];
    child.stderr.on("data", (chunk) => {
      const line = chunk.toString();
      stderrLines.push(line);
      if (stderrLines.length > 20) stderrLines.shift();
    });

    workerRl = readline.createInterface({ input: child.stdout });
    workerRl.on("line", (line) => {
      let parsed;
      try {
        parsed = JSON.parse(line);
      } catch {
        return;
      }
      const id = parsed?.id;
      if (!id || !pendingRequests.has(id)) return;
      const { resolve, reject, timer } = pendingRequests.get(id);
      clearTimeout(timer);
      pendingRequests.delete(id);
      if (parsed.ok) {
        resolve(parsed.result || {});
      } else {
        reject(new Error(parsed.error || "local asr worker failed"));
      }
    });

    child.on("error", (err) => {
      if (workerStartPromise) {
        workerStartPromise = null;
      }
      reject(err);
    });

    child.on("exit", (code) => {
      const err = new Error(`local asr worker exited with code ${code}`);
      for (const [, item] of pendingRequests) {
        clearTimeout(item.timer);
        item.reject(err);
      }
      pendingRequests.clear();
      asrWorker = null;
      workerRl = null;
      workerStartPromise = null;
    });

    setTimeout(() => {
      workerStartPromise = null;
      resolve();
    }, 100);
  });

  return workerStartPromise;
}

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

async function writeTempAudio(fileBuffer, originalName) {
  const safeName = String(originalName || "audio.webm").replace(/[^\w.\-]/g, "_");
  const tempPath = path.join(
    os.tmpdir(),
    `asr-${Date.now()}-${Math.random().toString(36).slice(2)}-${safeName}`
  );
  await fs.writeFile(tempPath, fileBuffer);
  return tempPath;
}

async function runLocalAsrScript(audioPath, languageHint) {
  await startAsrWorker();

  if (!asrWorker || asrWorker.killed || !asrWorker.stdin.writable) {
    throw new Error("local asr worker is not available");
  }

  const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pendingRequests.delete(id);
      reject(new Error("local asr timeout"));
    }, ASR_LOCAL_TIMEOUT_MS);

    pendingRequests.set(id, { resolve, reject, timer });
    asrWorker.stdin.write(
      `${JSON.stringify({ id, audio: audioPath, language: languageHint || ASR_LOCAL_LANGUAGE })}\n`
    );
  });
}

router.post("/transcribe", upload.single("file"), async (req, res, next) => {
  let tempAudioPath = "";
  try {
    getUserIdFromAuthHeader(req);

    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: "file is required (multipart/form-data, field: file)" });
    }

    const languageHintsRaw = req.body?.language_hints;
    const languageHint =
      typeof languageHintsRaw === "string" && languageHintsRaw.trim()
        ? languageHintsRaw.split(",").map((s) => s.trim()).filter(Boolean)[0]
        : ASR_LOCAL_LANGUAGE;

    tempAudioPath = await writeTempAudio(file.buffer, file.originalname);
    const result = await runLocalAsrScript(tempAudioPath, languageHint);
    const text = String(result?.text || "").trim();

    return res.json({
      provider: "local",
      model: `faster-whisper:${ASR_LOCAL_MODEL_SIZE}`,
      text,
      raw: result,
    });
  } catch (err) {
    next(err);
  } finally {
    if (tempAudioPath) {
      try {
        await fs.unlink(tempAudioPath);
      } catch {
        // ignore cleanup failure
      }
    }
  }
});

export default router;
