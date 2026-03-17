import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router";
import { Button, Card, CardBody, Chip, Textarea } from "@heroui/react";
import request from "../http/request";
import api from "../http/conservation";
import type { Message, Content } from "../types/conservation"

type ChatRole = "user" | "assistant";

type Conversation = {
  id: string;
  title: string;
  updatedAt: number;
};

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

// function normalizeMessage(raw: unknown): Message | null {
//   const data = asObject(raw);
//   const role = data.role === "assistant" ? "assistant" : data.role === "user" ? "user" : null;
//   const content = typeof data.content === "string" ? data.content : typeof data.message === "string" ? data.message : "";
//   if (!role || !content) return null;
//   return {
//     id: String(data.id ?? uid()),
//     role,
//     content,
//     createdAt: Number(data.createdAt ?? data.created_at ?? Date.now()),
//   };
// }

function normalizeConversation(raw: unknown, userId: string, index: number): Conversation | null {
  const data = asObject(raw);
  const id = data.id ?? data._id ?? (userId ? `${userId}-${index}` : "");
  if (!id) return null;
  // const rawMessages = Array.isArray(data.messages) ? data.messages : [];
  // const messages = rawMessages.map(normalizeMessage).filter(Boolean) as Message[];
  // console.log(data)
  const time = data.updated_at ?? data.updatedAt ?? data.created_at ?? Date.now();

  const updatedAt = typeof time === "number" ? time : new Date(time).getTime();
  return {
    id: String(id),
    title: typeof data.title === "string" && data.title.trim() ? data.title : "新对话",
    updatedAt,
  };
}

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function HomePage() {
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string>("");
  const messageEndRef = useRef<HTMLDivElement | null>(null);
  const [activeConversationMessages, setActiveConversationMessages] = useState<any[]>([]);
  const sortedConversations = useMemo(
    () => [...conversations].sort((a, b) => b.updatedAt - a.updatedAt),
    [conversations]
  );

  const loadConversations = async () => {
      try {
        const response = await api.Get_Conversation();
        const rawList = Array.isArray(response.data)
          ? response.data
          : Array.isArray(response.data?.conversations)
          ? response.data.conversations
          : [];
        const userId = getUserIdFromToken();
        const list = rawList.map((raw: unknown, index: number) => normalizeConversation(raw, userId, index)).filter(Boolean) as Conversation[];
        setConversations(list);
        console.log(list);
        getMessages(activeConversationId);
      } catch (error) {
        console.error("加载会话失败:", error);
      }
    };

  useEffect(() => {
    loadConversations();
  }, []);

  useEffect(() => {
    if (!activeConversationId && sortedConversations.length > 0) {
      setActiveConversationId(sortedConversations[0].id);
    }
  }, [activeConversationId, sortedConversations]);

  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeConversationId),
    [conversations, activeConversationId]
  );

  // useEffect(() => {
  //   messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  // }, [activeConversation?.messages.length, isSending]);

  const createNewConversation = async () => {
    try{
      const newConversation = {
        title: "新对话"
      }
      const response = await api.Create_Conversation(newConversation);
      loadConversations();
    }catch(error){
      console.error("创建对话失败", error)
    }
  };

  const getMessages = async (c_id: string) => {
    try{
      const response = await api.Get_Message(c_id);
      setActiveConversationMessages(response.data);
      console.log(activeConversationMessages);
    }catch(error){
      console.log(error);
    }
  };

  const updateConversationMessages = (id: string, updater: (messages: Message[]) => Message[]) => {
    const currentMessages = activeConversationMessages;
    const nextMessages = updater(currentMessages);
    setActiveConversationMessages(nextMessages as []);

    setConversations((prev) =>
      prev.map((conversation) => {
        if (conversation.id !== id) return conversation;
        return {
          ...conversation
        };
      })
    );
  };

  const submitMessage = async (e?: FormEvent) => {
    e?.preventDefault();
    try{
      const text = inputValue.trim();
      console.log(text);
      const content:Content = {
        text: text,
        attachments: []
      }
      const message:Message = {
        role: "user",
        content: content
      }
      const response = await api.Add_Message(message, activeConversationId);
    }catch {
      const fallbackMessage: Message = {
        role: "assistant",
        content: {text: "抱歉，服务暂时不可用，请稍后再试。", attachments: []}
      };
      updateConversationMessages(activeConversationId, (messages) => [...messages, fallbackMessage]);
    } finally {
      setIsSending(false);
    }
    
  }


  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    navigate("/login");
  };

  return (
    <div className="flex h-screen w-full  text-slate-800">
      <aside
        className={`${
          isSidebarOpen ? "w-[260px]" : "w-0"
        } overflow-hidden border-r border-slate-200 transition-all duration-300`}
      >
        <div className="flex h-full flex-col p-3">
          <Button
            onClick={createNewConversation}
            color="default"
            radius="lg"
            // className="bg-slate-900 text-sm font-semibold hover:bg-slate-700"
            className="border-0 data-[hover=true]:border-0 data-[hover=true]:shadow-none"
            variant="light"
          >
            + 新建对话
          </Button>

          <div className="mt-4 flex-1 space-y-2 overflow-y-auto">
            {sortedConversations.map((item) => {
              const isActive = item.id === activeConversationId;
              return (
                <Button
                  key={item.id}
                  onClick={() => {setActiveConversationId(item.id);getMessages(item.id);}}
                  variant={isActive ? "flat" : "light"}
                  className={`w-full justify-start rounded-lg border text-left text-sm transition-colors ${
                    isActive
                      ? "border-emerald-200 bg-emerald-50 font-semibold text-emerald-700 shadow-sm"
                      : "border-transparent hover:bg-white/70"
                  }`}
                >
                  {item.title}
                </Button>
              );
            })}
          </div>
          <Button onClick={handleLogout} variant="bordered" className="text-sm text-slate-600 hover:bg-white">
            退出登录
          </Button>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 items-center justify-between border-b border-slate-200 bg-white px-4">
          <div className="flex items-center gap-2">
            <Button
              onClick={() => setIsSidebarOpen((prev) => !prev)}
              variant="bordered"
              size="sm"
              className="border-slate-200"
            >
              菜单
            </Button>
            <span className="text-sm font-semibold">Chat Assistant</span>
          </div>
        </header>
        <div className="flex min-h-0 flex-1 flex-col">
          { <section className="mx-auto w-full max-w-4xl flex-1 overflow-y-auto px-4 py-6 md:px-6">
            {!activeConversationMessages.length ? (
              <div className="mt-14 text-center">
                <h1 className="text-3xl font-semibold text-slate-800">今天想聊点什么？</h1>
                <p className="mt-3 text-sm text-slate-500">我可以帮你写代码、改文案、解释概念或生成方案。</p>
                <div className="mx-auto mt-6 grid max-w-2xl grid-cols-1 gap-3 md:grid-cols-2">
                  {["帮我写一个 React 登录页", "解释下 JWT 登录流程", "生成一个课程学习计划", "优化这段 SQL 性能"].map(
                    (tip) => (
                      <Button
                        key={tip}
                        onClick={() => setInputValue(tip)}
                        variant="bordered"
                        className="h-auto justify-start whitespace-normal border-slate-200 bg-white p-4 text-left text-sm hover:border-emerald-300 hover:shadow-sm"
                      >
                        {tip}
                      </Button>
                    )
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {activeConversationMessages.map((message) => {
                  console.log(message);
                  const isUser = message.role === "user";
                  return (
                    <div key={message.id} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
                      <Card
                        shadow="none"
                        className={`max-w-[85%] ${
                          isUser ? "bg-[#F4F4F4] text-black" : "border border-slate-200 bg-white text-black"
                        }`}
                      >
                        <CardBody className="px-4 py-3 text-sm leading-7">{message.content.text}</CardBody>
                      </Card>
                    </div>
                  );
                })}
                {isSending ? (
                  <div className="flex justify-start">
                    <Card shadow="none" className="border border-slate-200 bg-white">
                      <CardBody className="px-4 py-3 text-sm text-slate-500">正在思考...</CardBody>
                    </Card>
                  </div>
                ) : null}
                <div ref={messageEndRef} />
              </div>
            )}
          </section> }

          <section className="border-t border-slate-200 bg-white px-4 py-4 md:px-6">
            <form onSubmit={submitMessage} className="mx-auto flex w-full max-w-4xl items-end gap-2">
              <div className="relative flex-1">
                <Textarea
                  minRows={1}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      submitMessage();
                    }
                  }}
                  placeholder="给 AI 发送消息..."
                  variant="bordered"
                  classNames={{
                    base: "w-full",
                    innerWrapper: "items-center",
                    input: "text-sm min-h-[28px] max-h-40 bg-slate-50 py-0 leading-6",
                  }}
                />
                
              </div>
              <Button
                type="submit"
                disabled={isSending || !inputValue.trim()}
                isDisabled={isSending || !inputValue.trim()}
                color="success"
                className="h-12 px-5 text-sm font-semibold"
              >
                发送
              </Button>
            </form>
          </section>
        </div>
      </main>
    </div>
  );
}
