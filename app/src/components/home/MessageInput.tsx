import type { FormEvent } from "react";
import { Button, Textarea } from "@heroui/react";

type MessageInputProps = {
  inputValue: string;
  isSending: boolean;
  onInputChange: (value: string) => void;
  onSubmit: (e?: FormEvent) => void;
};

export default function MessageInput({ inputValue, isSending, onInputChange, onSubmit }: MessageInputProps) {
  return (
    <section className="border-t border-slate-200 bg-white px-4 py-4 md:px-6">
      <form onSubmit={onSubmit} className="mx-auto flex w-full max-w-4xl items-end gap-2">
        <div className="relative flex-1">
          <Textarea
            minRows={1}
            value={inputValue}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSubmit();
              }
            }}
            placeholder="给 AI 发送消息..."
            variant="bordered"
            classNames={{
              base: "w-full",
              innerWrapper: "items-center",
              input: "min-h-[28px] max-h-40 bg-slate-50 py-0 text-sm leading-6",
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
  );
}
