const MODEL_SERVICE_URL = process.env.MODEL_SERVICE_URL || "http://127.0.0.1:8001";
const MODEL_SERVICE_TIMEOUT_MS = Number(process.env.MODEL_SERVICE_TIMEOUT_MS) || 300000;

export async function generateWithLocalModel({
  question,
  context,
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
