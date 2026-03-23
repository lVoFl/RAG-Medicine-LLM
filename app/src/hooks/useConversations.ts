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
      try {
        const text = inputValue.trim();
        if (!text || !activeConversationId) return;
        await api.SendAndGenerate({ question: text }, activeConversationId);
        setInputValue("");
        await getMessages(activeConversationId);
      } catch {
        const fallbackMessage: Message = {
          role: "assistant",
          content: { text: "抱歉，服务暂时不可用，请稍后再试。", attachments: [] },
        };
        updateConversationMessages(activeConversationId, (messages) => [...messages, fallbackMessage]);
      } finally {
        setIsSending(false);
      }
    },
    [activeConversationId, getMessages, inputValue, updateConversationMessages]
  );

  return {
    isSending,
    inputValue,
    setInputValue,
    activeConversationId,
    activeConversationMessages,
    sortedConversations,
    createNewConversation,
    handleSelectConversation,
    submitMessage,
  };
}
