import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import api from "../http/conservation";
import type { Message } from "../types/conservation";
import type { Conversation, ConversationMessage } from "../types/chat";

type RawData = Record<string, unknown>;

function asObject(value: unknown): RawData {
  return value && typeof value === "object" ? (value as RawData) : {};
}

function getUserIdFromToken() {
  const token = localStorage.getItem("token");
  if (!token) return "";

  const jwt = token.startsWith("Bearer ") ? token.slice(7) : token;
  const parts = jwt.split(".");
  if (parts.length < 2) return "";

  try {
    const payload = JSON.parse(atob(parts[1]));
    return String(payload?.id ?? payload?.userId ?? payload?.uid ?? "");
  } catch {
    return "";
  }
}

function normalizeConversation(raw: unknown, userId: string, index: number): Conversation | null {
  const data = asObject(raw);
  const id = data.id ?? data._id ?? (userId ? `${userId}-${index}` : "");
  if (!id) return null;

  const time = data.updated_at ?? data.updatedAt ?? data.created_at ?? Date.now();
  const updatedAt =
    typeof time === "number" || typeof time === "string" ? new Date(time).getTime() : Date.now();

  return {
    id: String(id),
    title: typeof data.title === "string" && data.title.trim() ? data.title : "新对话",
    updatedAt,
  };
}

export function useConversations() {
  const [isSending, setIsSending] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string>("");
  const [activeConversationMessages, setActiveConversationMessages] = useState<ConversationMessage[]>([]);

  const sortedConversations = useMemo(
    () => [...conversations].sort((a, b) => b.updatedAt - a.updatedAt),
    [conversations]
  );

  const getMessages = useCallback(async (conversationId: string) => {
    if (!conversationId) return;
    try {
      const response = await api.Get_Message(conversationId);
      const nextMessages = Array.isArray(response.data) ? response.data : [];
      setActiveConversationMessages(nextMessages as ConversationMessage[]);
    } catch (error) {
      console.error("加载消息失败:", error);
    }
  }, []);

  const loadConversations = useCallback(async () => {
    try {
      const response = await api.Get_Conversation();
      const rawList = Array.isArray(response.data)
        ? response.data
        : Array.isArray(response.data?.conversations)
        ? response.data.conversations
        : [];

      const userId = getUserIdFromToken();
      const list = rawList
        .map((raw: unknown, index: number) => normalizeConversation(raw, userId, index))
        .filter(Boolean) as Conversation[];

      setConversations(list);

      if (!list.length) {
        setActiveConversationId("");
        setActiveConversationMessages([]);
        return;
      }

      const nextActiveId = list[0].id;
      setActiveConversationId(nextActiveId);
      await getMessages(nextActiveId);
    } catch (error) {
      console.error("加载会话失败:", error);
    }
  }, [getMessages]);

  useEffect(() => {
    void loadConversations();
  }, [loadConversations]);

  const createNewConversation = useCallback(async () => {
    try {
      await api.Create_Conversation({ title: "新对话" });
      await loadConversations();
    } catch (error) {
      console.error("创建对话失败:", error);
    }
  }, [loadConversations]);

  const handleSelectConversation = useCallback(
    (conversationId: string) => {
      setActiveConversationId(conversationId);
      void getMessages(conversationId);
    },
    [getMessages]
  );

  const updateConversationMessages = useCallback(
    (id: string, updater: (messages: Message[]) => Message[]) => {
      setActiveConversationMessages((prev) => updater(prev as Message[]) as ConversationMessage[]);

      setConversations((prev) =>
        prev.map((conversation) => {
          if (conversation.id !== id) return conversation;
          return { ...conversation };
        })
      );
    },
    []
  );

  const submitMessage = useCallback(
    async (e?: FormEvent) => {
      e?.preventDefault();
      setIsSending(true);
      const conversationId = activeConversationId;
      const text = inputValue.trim();

      try {
        if (!text || !conversationId) return;

        const userMessage: Message = {
          role: "user",
          content: { text, attachments: [] },
        };
        const assistantPlaceholder: Message = {
          role: "assistant",
          content: { text: "", attachments: [] },
        };
        updateConversationMessages(conversationId, (messages) => [
          ...messages,
          userMessage,
          assistantPlaceholder,
        ]);

        setInputValue("");
        await api.SendAndGenerateStream(
          { question: text },
          conversationId,
          {
            onDelta: (delta) => {
              updateConversationMessages(conversationId, (messages) => {
                if (!messages.length) return messages;
                const nextMessages = [...messages];
                const lastIndex = nextMessages.length - 1;
                const lastMessage = nextMessages[lastIndex];
                if (lastMessage.role !== "assistant") return messages;
                const nextText = String(lastMessage.content?.text || "") + delta;
                nextMessages[lastIndex] = {
                  ...lastMessage,
                  content: {
                    ...(lastMessage.content || { attachments: [] }),
                    text: nextText,
                    attachments: Array.isArray(lastMessage.content?.attachments)
                      ? lastMessage.content.attachments
                      : [],
                  },
                };
                return nextMessages;
              });
            },
          }
        );

        await getMessages(conversationId);
      } catch {
        const fallbackMessage: Message = {
          role: "assistant",
          content: { text: "抱歉，服务暂时不可用，请稍后再试。", attachments: [] },
        };
        updateConversationMessages(conversationId, (messages) => {
          if (!messages.length) return [fallbackMessage];
          const nextMessages = [...messages];
          const lastIndex = nextMessages.length - 1;
          if (nextMessages[lastIndex].role === "assistant") {
            nextMessages[lastIndex] = fallbackMessage;
            return nextMessages;
          }
          return [...nextMessages, fallbackMessage];
        });
      } finally {
        setIsSending(false);
      }
    },
    [activeConversationId, getMessages, inputValue, updateConversationMessages]
  );

  const renameConversation = useCallback(async (conversationId: string, nextTitle: string) => {
    const title = nextTitle.trim();
    if (!conversationId || !title) return false;

    try {
      const response = await api.Patch_Conservation({ title }, conversationId);
      const updatedAtRaw = response.data?.updated_at ?? response.data?.updatedAt ?? Date.now();
      const updatedAt = new Date(updatedAtRaw).getTime() || Date.now();

      setConversations((prev) =>
        prev.map((conversation) =>
          conversation.id === conversationId ? { ...conversation, title, updatedAt } : conversation
        )
      );
      return true;
    } catch (error) {
      console.error("更新对话标题失败:", error);
      return false;
    }
  }, []);

  return {
    isSending,
    inputValue,
    setInputValue,
    activeConversationId,
    activeConversationMessages,
    sortedConversations,
    createNewConversation,
    renameConversation,
    handleSelectConversation,
    submitMessage,
  };
}
