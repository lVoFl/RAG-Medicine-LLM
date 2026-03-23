import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, MouseEvent } from "react";
import { useNavigate } from "react-router";
import { Button, Card, CardBody, Chip, Textarea, Input } from "@heroui/react";
import { Popover, PopoverTrigger, PopoverContent } from "@heroui/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import api from "../http/conservation";
import type { Message, Content } from "../types/conservation"
import { Edit, Trash } from "../icon/icon"

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
  const [openPopoverId, setOpenPopoverId] = useState<string | null>(null);
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
        const New_Id = list[0].id;
        setActiveConversationId(New_Id);
        getMessages(New_Id);
      } catch (error) {
        console.error("加载会话失败:", error);
      }
    };

  useEffect(() => {
    loadConversations();
  }, []);


  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeConversationId),
    [conversations, activeConversationId]
  );

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConversationMessages.length, isSending]);

  const createNewConversation = async () => {
    try{
      const newConversation = {
        title: "新对话"
      }
      const response = api.Create_Conversation(newConversation).then(() => {
        loadConversations();
      });
    }catch(error){
      console.error("创建对话失败", error)
    }
  };

  const getMessages = async (c_id: string) => {
    console.log(c_id);
    if (c_id === "") return;
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
    setIsSending(true);
    try{
      const text = inputValue.trim();
      if (!text || !activeConversationId) return;
      await api.SendAndGenerate({ question: text }, activeConversationId);
      setInputValue("");
      await getMessages(activeConversationId);
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

  const handleConversationRipple = (e: MouseEvent<HTMLDivElement>) => {
    const container = e.currentTarget;
    const rect = container.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height) * 1.2;
    const ripple = document.createElement("span");

    ripple.className = "conversation-ripple";
    ripple.style.width = `${size}px`;
    ripple.style.height = `${size}px`;
    ripple.style.left = `${e.clientX - rect.left}px`;
    ripple.style.top = `${e.clientY - rect.top}px`;

    container.appendChild(ripple);
    ripple.addEventListener("animationend", () => ripple.remove(), { once: true });
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
                <div
                  key={item.id}
                  onMouseDown={handleConversationRipple}
                  className={`conversation-ripple-container flex w-full items-center rounded-lg border text-sm transition-colors transition-transform duration-150 active:scale-[0.98] ${
                    isActive ? "bg-[#EAEAEA]" : "border-transparent hover:bg-[#EFEFEF]"
                  }`}
                >
                  <div
                    onClick={() => {setActiveConversationId(item.id);getMessages(item.id);}}
                    className="relative z-[1] flex-1 cursor-pointer px-3 py-2 text-left bg-transparent"
                  >
                    {item.title}
                  </div>
                  <Popover
                    placement="bottom"
                    isOpen={openPopoverId === item.id}
                    onOpenChange={(isOpen) => setOpenPopoverId(isOpen ? item.id : null)}
                  >
                    <PopoverTrigger>
                      <Button
                        isIconOnly
                        variant="light"
                        disableRipple
                        className="relative z-[1] ml-auto border-0 bg-transparent text-[#838383] shadow-none outline-none ring-0 data-[hover=true]:text-[#2E2E2E] data-[hover=true]:border-0 data-[hover=true]:border-transparent data-[hover=true]:shadow-none
                          data-[hover=true]:bg-transparent data-[pressed=true]:border-0 data-[pressed=true]:shadow-none data-[focus=true]:border-0 
                          data-[focus=true]:outline-none data-[focus=true]:ring-0 data-[focus=true]:shadow-none 
                          data-[focus-visible=true]:border-0 data-[focus-visible=true]:outline-none data-[focus-visible=true]:ring-0"
                      >
                        <Edit />
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[220px] rounded-lg border border-slate-200 bg-white p-1 shadow-md">
                      <div className="flex flex-col gap-1">
                        <Input
                          label="新标题"
                          size="sm"
                          variant="underlined"
                          classNames={{
                              label: "text-[12px] text-slate-500",
                              input: "text-sm",
                            }}
                          />
                          <div className="flex items-center justify-end gap-2">
                            <Button size="sm" color="primary" className="h-7 min-w-0 px-3 text-xs">
                              提交
                            </Button>
                            <Button
                              size="sm"
                              variant="faded"
                              className="h-7 min-w-0 px-2 text-xs text-slate-600"
                              onPress={() => setOpenPopoverId(null)}
                            >
                              取消
                            </Button>
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                  <Button
                    isIconOnly
                    variant="light"
                    disableRipple
                    className="relative z-[1] ml-auto border-0 bg-transparent text-[#838383] shadow-none outline-none ring-0 data-[hover=true]:text-[#2E2E2E] data-[hover=true]:border-0 data-[hover=true]:border-transparent data-[hover=true]:shadow-none
                      data-[hover=true]:bg-transparent data-[pressed=true]:border-0 data-[pressed=true]:shadow-none data-[focus=true]:border-0 
                      data-[focus=true]:outline-none data-[focus=true]:ring-0 data-[focus=true]:shadow-none 
                      data-[focus-visible=true]:border-0 data-[focus-visible=true]:outline-none data-[focus-visible=true]:ring-0"
                  >
                    <Trash />
                  </Button>
                </div>
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
                  const isUser = message.role === "user";
                  const markdownText = String(message?.content?.text ?? "");
                  return (
                    <div key={message.id} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
                      <Card
                        shadow="none"
                        className={`max-w-[85%] ${
                          isUser ? "bg-[#F4F4F4] text-black" : "border border-slate-200 bg-white text-black"
                        }`}
                      >
                        <CardBody className="px-4 py-3 text-sm leading-7">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                              ul: ({ children }) => <ul className="mb-2 list-disc pl-5 last:mb-0">{children}</ul>,
                              ol: ({ children }) => <ol className="mb-2 list-decimal pl-5 last:mb-0">{children}</ol>,
                              li: ({ children }) => <li className="mb-1">{children}</li>,
                              a: ({ href, children }) => (
                                <a
                                  href={href}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="text-emerald-700 underline break-all"
                                >
                                  {children}
                                </a>
                              ),
                              code: ({ className, children, ...props }) => {
                                const inline = !className;
                                return inline ? (
                                  <code className="rounded bg-slate-200 px-1 py-0.5 text-[0.9em]" {...props}>
                                    {children}
                                  </code>
                                ) : (
                                  <code className={className} {...props}>
                                    {children}
                                  </code>
                                );
                              },
                              pre: ({ children }) => (
                                <pre className="my-2 overflow-x-auto rounded-md bg-slate-900 p-3 text-slate-100">
                                  {children}
                                </pre>
                              ),
                            }}
                          >
                            {markdownText}
                          </ReactMarkdown>
                        </CardBody>
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
