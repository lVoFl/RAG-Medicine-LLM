import { useEffect, useRef, useState } from "react";
import type { MouseEvent } from "react";
import { useNavigate } from "react-router";
import { Button } from "@heroui/react";
import MessageInput from "../components/home/MessageInput";
import MessageList from "../components/home/MessageList";
import Sidebar from "../components/home/Sidebar";
import { useConversations } from "../hooks/useConversations";

export default function HomePage() {
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const messageEndRef = useRef<HTMLDivElement | null>(null);

  const {
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
  } = useConversations();

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConversationMessages.length, isSending]);

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
    <div className="flex h-screen w-full text-slate-800">
      <Sidebar
        isSidebarOpen={isSidebarOpen}
        conversations={sortedConversations}
        activeConversationId={activeConversationId}
        onCreateConversation={createNewConversation}
        onSelectConversation={handleSelectConversation}
        onRenameConversation={renameConversation}
        onLogout={handleLogout}
        onConversationMouseDown={handleConversationRipple}
      />

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
          <div className="min-h-0 flex-1 overflow-y-auto">
            <MessageList
              messages={activeConversationMessages}
              isSending={isSending}
              messageEndRef={messageEndRef}
              onSuggestionClick={setInputValue}
            />
          </div>
          <MessageInput
            inputValue={inputValue}
            isSending={isSending}
            onInputChange={setInputValue}
            onSubmit={submitMessage}
          />
        </div>
      </main>
    </div>
  );
}
