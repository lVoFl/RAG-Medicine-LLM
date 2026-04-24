import { useState } from "react";
import type { MutableRefObject } from "react";
import { Button, Card, CardBody } from "@heroui/react";
import { AnimatePresence, motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ConversationMessage } from "../../types/chat";
import type { RagDoc } from "../../types/conservation";

type MessageListProps = {
  messages: ConversationMessage[];
  isSending: boolean;
  messageEndRef: MutableRefObject<HTMLDivElement | null>;
  onSuggestionClick: (tip: string) => void;
};

const SUGGESTIONS = [
  "妊娠期糖尿病的血糖控制目标是什么？",
  "高血压危象的急诊处理流程有哪些关键步骤？",
  "2 型糖尿病患者如何制定饮食与运动计划？",
  "请总结糖尿病酮症酸中毒的诊断要点与处理原则",
];

function normalizeMarkdown(input: string): string {
  const text = (input || "").replace(/\r\n?/g, "\n");
  const unescapedNewline = text.includes("\\n") && !text.includes("\n") ? text.replace(/\\n/g, "\n") : text;
  // Some model outputs headings as `###标题`; add a space so markdown parser can recognize them.
  return unescapedNewline.replace(/(^|\n)(\s{0,3}#{1,6})([^\s#])/g, "$1$2 $3");
}

function normalizeRagDocs(raw: unknown): RagDoc[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const source = typeof item.source === "string" ? item.source : "";
      const headings = typeof item.headings === "string" ? item.headings : "";
      const content = typeof item.content === "string" ? item.content : "";
      if (!source && !headings && !content) return null;
      return {
        ...item,
        source,
        headings,
        content,
      } as RagDoc;
    })
    .filter(Boolean) as RagDoc[];
}

function getRagDocsFromMessage(message: ConversationMessage): RagDoc[] {
  const docsFromAttachments = normalizeRagDocs(message?.content?.attachments);
  if (docsFromAttachments.length) return dedupeBySource(docsFromAttachments);

  const docsFromContentField = normalizeRagDocs((message?.content as { retrieved_docs?: unknown[] })?.retrieved_docs);
  if (docsFromContentField.length) return dedupeBySource(docsFromContentField);

  return dedupeBySource(normalizeRagDocs((message as { retrieved_docs?: unknown[] })?.retrieved_docs));
}

function dedupeBySource(docs: RagDoc[]): RagDoc[] {
  const seen = new Set<string>();
  return docs.filter((doc) => {
    const key = String(doc.source || "未知来源").trim().toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

type CitationPanelProps = {
  ragDocs: RagDoc[];
};

function CitationPanel({ ragDocs }: CitationPanelProps) {
  const [open, setOpen] = useState(true);

  return (
    <div className="mt-3 border-t border-slate-200 pt-2 rag-citations">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full cursor-pointer items-center gap-2 text-left text-sm font-medium text-slate-700"
      >
        <span
          className={`inline-block text-slate-500 transition-transform duration-200 ${open ? "rotate-90" : "rotate-0"}`}
        >
          ☰
        </span>
        <span>全部引用</span>
        <span className="text-xs text-slate-400">({ragDocs.length})</span>
      </button>

      <AnimatePresence initial={false}>
        {open ? (
          <motion.div
            key="citation-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.24, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <ol className="mt-2 space-y-2 text-sm text-slate-700">
              {ragDocs.map((doc, docIndex) => (
                <li key={`${doc.chunk_id ?? doc.source ?? "doc"}-${docIndex}`} className="leading-6">
                  <span className="mr-2 text-slate-500">{docIndex + 1}.</span>
                  <span>{doc.source || "未知来源"}</span>
                </li>
              ))}
            </ol>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
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
          <p className="mt-3 text-sm text-slate-500">我可以帮你解读医学知识、总结指南要点并生成健康管理建议。</p>
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
            const ragDocs = isUser ? [] : getRagDocsFromMessage(message);
            if (!isUser && isSending && !markdownText.trim()) {
              return null;
            }
            const key = message.id ?? `${message.role}-${index}`;
            return isUser ? (
              <div key={key} className="flex justify-end">
                <Card shadow="none" className="max-w-[85%] bg-[#F4F4F4] text-black">
                  <CardBody className="px-4 py-3 text-base leading-8">
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
            ) : (
              <div key={key} className="text-black">
                <div className="text-base leading-8">
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
                  {ragDocs.length ? <CitationPanel ragDocs={ragDocs} /> : null}
                </div>
              </div>
            );
          })}
          {showThinking ? (
            <div className="text-base text-slate-500">正在思考...</div>
          ) : null}
          <div ref={messageEndRef} />
        </div>
      )}
    </section>
  );
}
