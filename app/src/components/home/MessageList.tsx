import type { MutableRefObject } from "react";
import { Button, Card, CardBody } from "@heroui/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ConversationMessage } from "../../types/chat";

type MessageListProps = {
  messages: ConversationMessage[];
  isSending: boolean;
  messageEndRef: MutableRefObject<HTMLDivElement | null>;
  onSuggestionClick: (tip: string) => void;
};

const SUGGESTIONS = ["帮我写一个 React 登录页", "解释下 JWT 登录流程", "生成一个课程学习计划", "优化这段 SQL 性能"];

function normalizeMarkdown(input: string): string {
  const text = (input || "").replace(/\r\n?/g, "\n");
  const unescapedNewline = text.includes("\\n") && !text.includes("\n") ? text.replace(/\\n/g, "\n") : text;
  // Some model outputs headings as `###标题`; add a space so markdown parser can recognize them.
  return unescapedNewline.replace(/(^|\n)(\s{0,3}#{1,6})([^\s#])/g, "$1$2 $3");
}

export default function MessageList({ messages, isSending, messageEndRef, onSuggestionClick }: MessageListProps) {
  const lastMessage = messages.length ? messages[messages.length - 1] : null;
  const lastAssistantText =
    lastMessage?.role === "assistant" ? String(lastMessage?.content?.text ?? "").trim() : "";
  const showThinking = isSending && !lastAssistantText;

  return (
    <section className="mx-auto w-full max-w-4xl flex-1 px-4 py-6 md:px-6">
      {!messages.length ? (
        <div className="mt-14 text-center">
          <h1 className="text-3xl font-semibold text-slate-800">今天想聊点什么？</h1>
          <p className="mt-3 text-sm text-slate-500">我可以帮你写代码、改文案、解释概念或生成方案。</p>
          <div className="mx-auto mt-6 grid max-w-2xl grid-cols-1 gap-3 md:grid-cols-2">
            {SUGGESTIONS.map((tip) => (
              <Button
                key={tip}
                onClick={() => onSuggestionClick(tip)}
                variant="bordered"
                className="h-auto justify-start whitespace-normal border-slate-200 bg-white p-4 text-left text-sm hover:border-emerald-300 hover:shadow-sm"
              >
                {tip}
              </Button>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {messages.map((message, index) => {
            const isUser = message.role === "user";
            const markdownText = normalizeMarkdown(String(message?.content?.text ?? ""));
            if (!isUser && isSending && !markdownText.trim()) {
              return null;
            }
            const key = message.id ?? `${message.role}-${index}`;
            return (
              <div key={key} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
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
                        h1: ({ children }) => <h1 className="mb-3 mt-4 text-2xl font-bold leading-8 first:mt-0">{children}</h1>,
                        h2: ({ children }) => <h2 className="mb-3 mt-4 text-xl font-semibold leading-7 first:mt-0">{children}</h2>,
                        h3: ({ children }) => <h3 className="mb-2 mt-3 text-lg font-semibold leading-7 first:mt-0">{children}</h3>,
                        h4: ({ children }) => <h4 className="mb-2 mt-3 text-base font-semibold leading-6 first:mt-0">{children}</h4>,
                        h5: ({ children }) => <h5 className="mb-2 mt-2 text-sm font-semibold leading-6 first:mt-0">{children}</h5>,
                        h6: ({ children }) => <h6 className="mb-2 mt-2 text-sm font-medium leading-6 text-slate-600 first:mt-0">{children}</h6>,
                        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                        ul: ({ children }) => <ul className="mb-2 list-disc pl-5 last:mb-0">{children}</ul>,
                        ol: ({ children }) => <ol className="mb-2 list-decimal pl-5 last:mb-0">{children}</ol>,
                        li: ({ children }) => <li className="mb-1">{children}</li>,
                        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                        a: ({ href, children }) => (
                          <a href={href} target="_blank" rel="noreferrer" className="break-all text-emerald-700 underline">
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
                          <pre className="my-2 overflow-x-auto rounded-md bg-slate-900 p-3 text-slate-100">{children}</pre>
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
          {showThinking ? (
            <div className="flex justify-start">
              <Card shadow="none" className="border border-slate-200 bg-white">
                <CardBody className="px-4 py-3 text-sm text-slate-500">正在思考...</CardBody>
              </Card>
            </div>
          ) : null}
          <div ref={messageEndRef} />
        </div>
      )}
    </section>
  );
}
