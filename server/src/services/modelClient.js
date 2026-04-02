const MODEL_SERVICE_URL = process.env.MODEL_SERVICE_URL || "http://127.0.0.1:8001";
const MODEL_SERVICE_TIMEOUT_MS = Number(process.env.MODEL_SERVICE_TIMEOUT_MS) || 300000;

export async function generateWithLocalModel({
  question,
  context,
  history,
  systemPrompt,
  maxNewTokens,
  temperature,
  topP,
}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), MODEL_SERVICE_TIMEOUT_MS);

  try {
    const response = await fetch(`${MODEL_SERVICE_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      signal: controller.signal,
      body: JSON.stringify({
        question,
        context,
        history,
        system_prompt: systemPrompt,
        max_new_tokens: maxNewTokens,
        temperature,
        top_p: topP,
      }),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const message = data.error || `model service returned ${response.status}`;
      const error = new Error(message);
      error.status = 502;
      throw error;
    }

    return data;
  } catch (err) {
    if (err.name === "AbortError") {
      const error = new Error("model service request timed out");
      error.status = 504;
      throw error;
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export async function generateWithLocalModelStream({
  question,
  context,
  history,
  systemPrompt,
  maxNewTokens,
  temperature,
  topP,
}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), MODEL_SERVICE_TIMEOUT_MS);

  try {
    const response = await fetch(`${MODEL_SERVICE_URL}/generate/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      signal: controller.signal,
      body: JSON.stringify({
        question,
        context,
        history,
        system_prompt: systemPrompt,
        max_new_tokens: maxNewTokens,
        temperature,
        top_p: topP,
      }),
    });

    if (!response.ok || !response.body) {
      const data = await response.json().catch(() => ({}));
      const message = data.error || `model service returned ${response.status}`;
      const error = new Error(message);
      error.status = response.status === 408 ? 504 : 502;
      throw error;
    }

    return {
      response,
      abort: () => controller.abort(),
      clearTimeout: () => clearTimeout(timeout),
    };
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === "AbortError") {
      const error = new Error("model service request timed out");
      error.status = 504;
      throw error;
    }
    throw err;
  }
}
