import { createMessage, ensureConversationAccessible, patchMessage } from "./messageService.js";
import { generateWithLocalModel } from "./modelClient.js";

function normalizeUsage(usage) {
  if (!usage || typeof usage !== "object") return null;
  const promptTokens = Number(usage.prompt_tokens || 0);
  const completionTokens = Number(usage.completion_tokens || 0);
  return {
    ...usage,
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  };
}

export async function runConversationWorkflow({
  userId,
  conversationId,
  question,
  context,
  systemPrompt,
  maxNewTokens,
  temperature,
  topP,
}) {
  await ensureConversationAccessible(conversationId, userId);

  const userMessage = await createMessage({
    conversationId,
    role: "user",
    content: { text: question, attachments: [] },
    tokens: null,
  });

  const modelResult = await generateWithLocalModel({
    question,
    context,
    systemPrompt,
    maxNewTokens,
    temperature,
    topP,
  });

  const usage = normalizeUsage(modelResult.usage);
  let persistedUserMessage = userMessage;
  if (usage?.prompt_tokens != null) {
    persistedUserMessage = await patchMessage({
      conversationId,
      messageId: userMessage.id,
      tokens: usage.prompt_tokens,
    });
  }

  const assistantMessage = await createMessage({
    conversationId,
    role: "assistant",
    content: { 
      text: String(modelResult.answer || ""), 
      usage: usage
    },
    tokens: usage?.completion_tokens ?? null,
  });

  return {
    user_message: persistedUserMessage,
    assistant_message: assistantMessage,
    answer: assistantMessage.content.text,
    usage,
    params: modelResult.params || null,
  };
}
