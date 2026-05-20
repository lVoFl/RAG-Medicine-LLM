import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type MarkdownRendererProps = {
  text: string;
};

export default function MarkdownRenderer({ text }: MarkdownRendererProps) {
  return (
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
      {text}
    </ReactMarkdown>
  );
}
