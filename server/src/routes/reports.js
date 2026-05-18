import { Router } from "express";
import jwt from "jsonwebtoken";
import crypto from "crypto";
import multer from "multer";
import JSZip from "jszip";

const router = Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
});

const JWT_SECRET = process.env.JWT_SECRET || "change-this-secret-in-production";
const MINERU_API_BASE = (process.env.MINERU_API_BASE || "https://mineru.net/api/v4").replace(/\/+$/, "");
const MINERU_POLL_INTERVAL_MS = Number(process.env.MINERU_POLL_INTERVAL_MS || 3000);
const MINERU_POLL_TIMEOUT_MS = Number(process.env.MINERU_POLL_TIMEOUT_MS || 180000);

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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function walkValues(input, visitor) {
  if (Array.isArray(input)) {
    input.forEach((item) => walkValues(item, visitor));
    return;
  }
  if (!input || typeof input !== "object") return;
  for (const [key, value] of Object.entries(input)) {
    visitor(key, value);
    walkValues(value, visitor);
  }
}

function collectStates(payload) {
  const states = [];
  walkValues(payload, (key, value) => {
    if (key.toLowerCase() === "state" && typeof value === "string") {
      states.push(value.toLowerCase());
    }
  });
  return states;
}

function findFirstUrl(payload, predicate) {
  let found = "";
  walkValues(payload, (_key, value) => {
    if (found) return;
    if (typeof value === "string" && /^https?:\/\//i.test(value) && predicate(value)) {
      found = value;
    }
  });
  return found;
}

async function fetchBatchResult(mineruApiKey, batchId) {
  const resp = await fetch(`${MINERU_API_BASE}/extract-results/batch/${batchId}`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${mineruApiKey}`,
      Accept: "*/*",
    },
  });
  const raw = await resp.text();
  let data;
  try {
    data = raw ? JSON.parse(raw) : null;
  } catch {
    data = { raw };
  }
  return { ok: resp.ok, status: resp.status, data };
}

async function extractTextFromZipUrl(zipUrl) {
  const zipResp = await fetch(zipUrl);
  if (!zipResp.ok) {
    throw new Error(`download zip failed: ${zipResp.status}`);
  }
  const zipBuffer = Buffer.from(await zipResp.arrayBuffer());
  const zip = await JSZip.loadAsync(zipBuffer);
  const textChunks = [];

  const entries = Object.values(zip.files)
    .filter((entry) => !entry.dir)
    .filter((entry) => /\.(md|markdown|txt)$/i.test(entry.name));

  for (const entry of entries) {
    const content = await entry.async("string");
    if (content && content.trim()) {
      textChunks.push(content.trim());
    }
  }

  return textChunks.join("\n\n");
}

// router.post("/mineru/upload-pdf", upload.single("file"), async (req, res, next) => {
//   try {
//     getUserIdFromAuthHeader(req);

//     const mineruApiKey = process.env.MINERU_API_KEY;
//     if (!mineruApiKey) {
//       return res.status(500).json({ error: "MINERU_API_KEY is not configured" });
//     }

//     const file = req.file;
//     if (!file) {
//       return res.status(400).json({ error: "file is required (multipart/form-data, field: file)" });
//     }

//     const fileName = String(file.originalname || "report.pdf").trim() || "report.pdf";
//     if (!fileName.toLowerCase().endsWith(".pdf")) {
//       return res.status(400).json({ error: "only pdf file is supported" });
//     }

//     const modelVersion =
//       typeof req.body?.model_version === "string" && req.body.model_version.trim()
//         ? req.body.model_version.trim()
//         : "vlm";
//     const dataId =
//       typeof req.body?.data_id === "string" && req.body.data_id.trim()
//         ? req.body.data_id.trim()
//         : `report_${Date.now()}_${crypto.randomUUID()}`;

//     const createBatchResp = await fetch(`${MINERU_API_BASE}/file-urls/batch`, {
//       method: "POST",
//       headers: {
//         Authorization: `Bearer ${mineruApiKey}`,
//         "Content-Type": "application/json",
//         Accept: "*/*",
//       },
//       body: JSON.stringify({
//         files: [{ name: fileName, data_id: dataId }],
//         model_version: modelVersion,
//       }),
//     });

//     const createBatchRaw = await createBatchResp.text();
//     let createBatchData;
//     try {
//       createBatchData = createBatchRaw ? JSON.parse(createBatchRaw) : null;
//     } catch {
//       createBatchData = { raw: createBatchRaw };
//     }

//     if (!createBatchResp.ok) {
//       return res.status(createBatchResp.status).json({
//         provider: "mineru",
//         step: "create_batch",
//         error: "failed to create batch upload url",
//         data: createBatchData,
//       });
//     }

//     const uploadUrl = createBatchData?.data?.file_urls?.[0];
//     const batchId = createBatchData?.data?.batch_id;
//     if (!uploadUrl || !batchId) {
//       return res.status(502).json({
//         provider: "mineru",
//         step: "create_batch",
//         error: "invalid mineru response: missing upload url or batch id",
//         data: createBatchData,
//       });
//     }

//     const uploadResp = await fetch(uploadUrl, {
//       method: "PUT",
//       body: file.buffer,
//     });

//     if (!uploadResp.ok) {
//       const uploadRaw = await uploadResp.text();
//       return res.status(uploadResp.status).json({
//         provider: "mineru",
//         step: "upload_pdf",
//         error: "failed to upload file to mineru storage",
//         batch_id: batchId,
//         data_id: dataId,
//         data: uploadRaw || null,
//       });
//     }

//     res.json({
//       provider: "mineru",
//       endpoint: "/upload-pdf",
//       message: "pdf uploaded to mineru successfully",
//       batch_id: batchId,
//       submitted_file: {
//         name: fileName,
//         data_id: dataId,
//         size: file.size,
//       },
//       next_step: "use batch_id to trigger/track parsing in MinerU",
//     });
//   } catch (err) {
//     next(err);
//   }
// });

router.post("/mineru/upload-pdf/extract-text", upload.single("file"), async (req, res, next) => {
  try {
    getUserIdFromAuthHeader(req);

    const mineruApiKey = process.env.MINERU_API_KEY;
    if (!mineruApiKey) {
      return res.status(500).json({ error: "MINERU_API_KEY is not configured" });
    }

    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: "file is required (multipart/form-data, field: file)" });
    }

    const fileName = String(file.originalname || "report.pdf").trim() || "report.pdf";
    if (!fileName.toLowerCase().endsWith(".pdf")) {
      return res.status(400).json({ error: "only pdf file is supported" });
    }

    const modelVersion =
      typeof req.body?.model_version === "string" && req.body.model_version.trim()
        ? req.body.model_version.trim()
        : "vlm";
    const dataId =
      typeof req.body?.data_id === "string" && req.body.data_id.trim()
        ? req.body.data_id.trim()
        : `report_${Date.now()}_${crypto.randomUUID()}`;

    const createBatchResp = await fetch(`${MINERU_API_BASE}/file-urls/batch`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${mineruApiKey}`,
        "Content-Type": "application/json",
        Accept: "*/*",
      },
      body: JSON.stringify({
        files: [{ name: fileName, data_id: dataId }],
        model_version: modelVersion,
      }),
    });

    const createBatchRaw = await createBatchResp.text();
    let createBatchData;
    try {
      createBatchData = createBatchRaw ? JSON.parse(createBatchRaw) : null;
    } catch {
      createBatchData = { raw: createBatchRaw };
    }

    if (!createBatchResp.ok) {
      return res.status(createBatchResp.status).json({
        provider: "mineru",
        step: "create_batch",
        error: "failed to create batch upload url",
        data: createBatchData,
      });
    }

    const uploadUrl = createBatchData?.data?.file_urls?.[0];
    const batchId = createBatchData?.data?.batch_id;
    if (!uploadUrl || !batchId) {
      return res.status(502).json({
        provider: "mineru",
        step: "create_batch",
        error: "invalid mineru response: missing upload url or batch id",
        data: createBatchData,
      });
    }

    const uploadResp = await fetch(uploadUrl, {
      method: "PUT",
      body: file.buffer,
    });
    if (!uploadResp.ok) {
      const uploadRaw = await uploadResp.text();
      return res.status(uploadResp.status).json({
        provider: "mineru",
        step: "upload_pdf",
        error: "failed to upload file to mineru storage",
        batch_id: batchId,
        data_id: dataId,
        data: uploadRaw || null,
      });
    }

    const startedAt = Date.now();
    let lastResult = null;
    while (Date.now() - startedAt < MINERU_POLL_TIMEOUT_MS) {
      const queryResult = await fetchBatchResult(mineruApiKey, batchId);
      lastResult = queryResult;
      if (!queryResult.ok) {
        return res.status(queryResult.status).json({
          provider: "mineru",
          step: "query_result",
          error: "failed to query batch result",
          batch_id: batchId,
          data_id: dataId,
          data: queryResult.data,
        });
      }

      const states = collectStates(queryResult.data);
      const hasFailed = states.some((s) => s.includes("fail"));
      if (hasFailed) {
        return res.status(502).json({
          provider: "mineru",
          step: "parse_failed",
          error: "mineru parse task failed",
          batch_id: batchId,
          data_id: dataId,
          data: queryResult.data,
        });
      }

      const hasRunning = states.some((s) => ["pending", "running", "uploading", "waiting-file"].includes(s));
      if (hasRunning) {
        await sleep(MINERU_POLL_INTERVAL_MS);
        continue;
      }

      const markdownUrl = findFirstUrl(
        queryResult.data,
        (url) => /\.md($|\?)/i.test(url) || /markdown/i.test(url)
      );
      if (markdownUrl) {
        const mdResp = await fetch(markdownUrl);
        if (!mdResp.ok) {
          throw new Error(`download markdown failed: ${mdResp.status}`);
        }
        const text = await mdResp.text();
        return res.json({
          provider: "mineru",
          endpoint: "/upload-pdf/extract-text",
          message: "pdf parsed successfully",
          batch_id: batchId,
          data_id: dataId,
          text: text || "",
          result_meta: queryResult.data,
        });
      }

      const zipUrl = findFirstUrl(queryResult.data, (url) => /\.zip($|\?)/i.test(url));
      if (zipUrl) {
        const text = await extractTextFromZipUrl(zipUrl);
        return res.json({
          provider: "mineru",
          endpoint: "/upload-pdf/extract-text",
          message: "pdf parsed successfully",
          batch_id: batchId,
          data_id: dataId,
          text: text || "",
          result_meta: queryResult.data,
        });
      }

      return res.json({
        provider: "mineru",
        endpoint: "/upload-pdf/extract-text",
        message: "task finished but no markdown/zip url found",
        batch_id: batchId,
        data_id: dataId,
        result_meta: queryResult.data,
      });
    }

    return res.status(202).json({
      provider: "mineru",
      endpoint: "/upload-pdf/extract-text",
      message: "poll timeout, task still processing",
      batch_id: batchId,
      data_id: dataId,
      result_meta: lastResult?.data || null,
    });
  } catch (err) {
    next(err);
  }
});

export default router;
