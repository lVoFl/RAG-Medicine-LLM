import {
  createMessage,
  ensureConversationAccessible,
  listMessagesByConversation,
  patchMessage,
} from "./messageService.js";
import { generateWithLocalModel } from "./modelClient.js";

const MODEL_PROMPT_MAX_TOKENS = Number(process.env.MODEL_PROMPT_MAX_TOKENS) || 4096;
const MODEL_RESPONSE_RESERVE_TOKENS =
  Number(process.env.MODEL_RESPONSE_RESERVE_TOKENS) || 1024;
const MODEL_HISTORY_MAX_TOKENS = Number(process.env.MODEL_HISTORY_MAX_TOKENS) || 2048;
const PROMPT_SAFETY_MARGIN_TOKENS = Number(process.env.MODEL_PROMPT_SAFETY_MARGIN_TOKENS) || 128;

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

function extractMessageText(content) {
  if (content == null) return "";
  if (typeof content === "string") return content.trim();
  if (typeof content === "object") {
    if (typeof content.text === "string") return content.text.trim();
    return JSON.stringify(content);
  }
  return String(content).trim();
}

function estimateTokens(text) {
  const value = String(text || "");
  if (!value.trim()) return 0;

  const cjkMatches = value.match(/[\u3400-\u9FFF]/g);
  const cjkCount = cjkMatches ? cjkMatches.length : 0;
  const punctuationMatches = value.match(/[^\w\s]/g);
  const punctuationCount = punctuationMatches ? punctuationMatches.length : 0;
  const nonCjkLength = value.length - cjkCount;

  const nonCjkTokens = Math.ceil(nonCjkLength / 4);
  const cjkTokens = cjkCount;
  const punctuationTokens = Math.ceil(punctuationCount * 0.25);
  return nonCjkTokens + cjkTokens + punctuationTokens;
}

function buildConversationHistoryWithinTokenBudget(historyMessages, maxTokens) {
  if (!Array.isArray(historyMessages) || historyMessages.length === 0 || maxTokens <= 0) {
    return { history: [], usedTokens: 0, includedMessages: 0 };
  }

  const selectedMessages = [];
  let usedTokens = 0;

  for (let index = historyMessages.length - 1; index >= 0; index -= 1) {
    const message = historyMessages[index];
    const text = extractMessageText(message.content);
    if (!text) continue;

    const rawRole = String(message.role || "user").toLowerCase();
    const normalizedRole = rawRole === "assistant" ? "assistant" : "user";
    const lineTokens = estimateTokens(text) + 4;

    if (usedTokens + lineTokens > maxTokens) {
      break;
    }

    usedTokens += lineTokens;
    selectedMessages.unshift({
      role: normalizedRole,
      content: text,
    });
  }

  return {
    history: selectedMessages,
    usedTokens,
    includedMessages: selectedMessages.length,
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
  const historyMessages = await listMessagesByConversation(conversationId);

  const baseContext = String(context || "").trim();
  const questionTokens = estimateTokens(question);
  const systemPromptTokens = estimateTokens(systemPrompt);
  const baseContextTokens = estimateTokens(baseContext);
  const parsedMaxNewTokens = Number(maxNewTokens);
  const responseReserveTokens =
    Number.isFinite(parsedMaxNewTokens) && parsedMaxNewTokens > 0
      ? parsedMaxNewTokens
      : MODEL_RESPONSE_RESERVE_TOKENS;

  const remainingPromptBudget = Math.max(
    0,
    MODEL_PROMPT_MAX_TOKENS -
      responseReserveTokens -
      questionTokens -
      systemPromptTokens -
      baseContextTokens -
      PROMPT_SAFETY_MARGIN_TOKENS
  );

  const historyTokenBudget = Math.min(MODEL_HISTORY_MAX_TOKENS, remainingPromptBudget);
  const { history } = buildConversationHistoryWithinTokenBudget(
    historyMessages,
    historyTokenBudget
  );

  const userMessage = await createMessage({
    conversationId,
    role: "user",
    content: { text: question, attachments: [] },
    tokens: null,
  });

  const modelResult = await generateWithLocalModel({
    question,
    context: baseContext || undefined,
    history: history.length > 0 ? history : undefined,
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
