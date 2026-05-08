import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import api from "../http/conservation";
import type { Message, RagDoc } from "../types/conservation";
import type { Conversation, ConversationMessage } from "../types/chat";

type RawData = Record<string, unknown>;

function asObject(value: unknown): RawData {
  return value && typeof value === "object" ? (value as RawData) : {};
}

function normalizeRagDocs(value: unknown): RagDoc[] {
  if (!Array.isArray(value)) return [];

  return value
    .map((item) => {
      const data = asObject(item);
      const source = typeof data.source === "string" ? data.source : "";
      const headings = typeof data.headings === "string" ? data.headings : "";
      const content = typeof data.content === "string" ? data.content : "";
      if (!source && !headings && !content) return null;

      return {
        ...data,
        source,
        headings,
        content,
      } as RagDoc;
    })
    .filter(Boolean) as RagDoc[];
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
    messageCount:
      typeof data.message_count === "number"
        ? data.message_count
        : typeof data.messageCount === "number"
        ? data.messageCount
        : 0,
  };
}

function isEmptyNewConversation(conversation: Conversation): boolean {
  return conversation.title.trim() === "新对话" && conversation.messageCount === 0;
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

      const emptyNewConversations = list.filter(isEmptyNewConversation);
      if (emptyNewConversations.length > 1) {
        const [keep, ...toDelete] = emptyNewConversations.sort((a, b) => b.updatedAt - a.updatedAt);
        void Promise.allSettled(toDelete.map((conversation) => api.Delete_Conversation(conversation.id)));
        const deletedIds = new Set(toDelete.map((conversation) => conversation.id));
        const dedupedList = list.filter((conversation) => !deletedIds.has(conversation.id) || conversation.id === keep.id);
        setConversations(dedupedList);

        if (!dedupedList.length) {
          setActiveConversationId("");
          setActiveConversationMessages([]);
          return;
        }

        const nextActiveId = dedupedList[0].id;
        setActiveConversationId(nextActiveId);
        await getMessages(nextActiveId);
        return;
      }

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
      const existingEmptyConversation = sortedConversations.find(isEmptyNewConversation);
      if (existingEmptyConversation) {
        setActiveConversationId(existingEmptyConversation.id);
        await getMessages(existingEmptyConversation.id);
        return;
      }

      await api.Create_Conversation({ title: "新对话" });
      await loadConversations();
    } catch (error) {
      console.error("创建对话失败:", error);
    }
  }, [getMessages, loadConversations, sortedConversations]);

  const handleSelectConversation = useCallback(
    (conversationId: string) => {
      setActiveConversationId(conversationId);
      void getMessages(conversationId);
    },
    [getMessages]
  );

  const updateConversationMessages = useCallback(
    (id: string, updater: (messages: Message[]) => Message[]) => {
      setActiveConversationMessages((prev) => {
        const next = updater(prev as Message[]) as ConversationMessage[];
        return next;
      });

      setConversations((prev) =>
        prev.map((conversation) => {
          if (conversation.id !== id) return conversation;
          const isFirstMessage = conversation.messageCount === 0;
          return { ...conversation, messageCount: isFirstMessage ? 1 : conversation.messageCount };
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

        const activeConversation = conversations.find((conversation) => conversation.id === conversationId);
        const shouldAutoRename =
          activeConversation?.title.trim() === "新对话" && activeConversationMessages.length === 0;

        if (shouldAutoRename) {
          const updatedAt = Date.now();
          setConversations((prev) =>
            prev.map((conversation) =>
              conversation.id === conversationId ? { ...conversation, title: text, updatedAt } : conversation
            )
          );

          void api.Patch_Conservation({ title: text }, conversationId).catch((error) => {
            console.error("自动更新会话标题失败:", error);
          });
        }

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
            onDone: (event) => {
              if (event.type !== "end") return;
              const docs = normalizeRagDocs(event.retrieved_docs);
              console.log(docs)
              if (!docs.length) return;

              updateConversationMessages(conversationId, (messages) => {
                if (!messages.length) return messages;
                const nextMessages = [...messages];
                const lastIndex = nextMessages.length - 1;
                const lastMessage = nextMessages[lastIndex];
                if (lastMessage.role !== "assistant") return messages;

                nextMessages[lastIndex] = {
                  ...lastMessage,
                  content: {
                    ...(lastMessage.content || { text: "", attachments: [] }),
                    attachments: docs,
                    retrieved_docs: docs,
                    params: event.params ?? lastMessage.content?.params,
                    usage: event.usage ?? lastMessage.content?.usage,
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
    [
      activeConversationId,
      activeConversationMessages.length,
      conversations,
      getMessages,
      inputValue,
      updateConversationMessages,
    ]
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

  const deleteConversation = useCallback(
    async (conversationId: string) => {
      if (!conversationId) return false;

      try {
        await api.Delete_Conversation(conversationId);

        const nextConversations = conversations.filter((conversation) => conversation.id !== conversationId);
        setConversations(nextConversations);

        if (activeConversationId !== conversationId) return true;

        if (!nextConversations.length) {
          setActiveConversationId("");
          setActiveConversationMessages([]);
          return true;
        }

        const nextActiveConversation = [...nextConversations].sort((a, b) => b.updatedAt - a.updatedAt)[0];
        setActiveConversationId(nextActiveConversation.id);
        await getMessages(nextActiveConversation.id);
        return true;
      } catch (error) {
        console.error("删除会话失败:", error);
        return false;
      }
    },
    [activeConversationId, conversations, getMessages]
  );

  useEffect(() => {
    const cleanupEmptyConversations = () => {
      const emptyIds = conversations.filter(isEmptyNewConversation).map((conversation) => conversation.id);
      if (!emptyIds.length) return;
      emptyIds.forEach((conversationId) => {
        void api.Delete_Conversation(conversationId).catch((error) => {
          console.error("退出时清理空会话失败:", error);
        });
      });
    };

    const handlePageHide = () => {
      cleanupEmptyConversations();
    };

    window.addEventListener("pagehide", handlePageHide);
    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      cleanupEmptyConversations();
    };
  }, [conversations]);

  return {
    isSending,
    inputValue,
    setInputValue,
    activeConversationId,
    activeConversationMessages,
    sortedConversations,
    createNewConversation,
    renameConversation,
    deleteConversation,
    handleSelectConversation,
    submitMessage,
  };
}
