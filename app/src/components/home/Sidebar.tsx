import { useState } from "react";
import type { MouseEvent } from "react";
import { Button, Input, Popover, PopoverContent, PopoverTrigger } from "@heroui/react";
import { Edit, Trash } from "../../icon/icon";
import type { Conversation } from "../../types/chat";

type SidebarProps = {
  isSidebarOpen: boolean;
  conversations: Conversation[];
  activeConversationId: string;
  onCreateConversation: () => void;
  onSelectConversation: (id: string) => void;
  onRenameConversation: (id: string, title: string) => Promise<boolean>;
  onLogout: () => void;
  onConversationMouseDown: (e: MouseEvent<HTMLDivElement>) => void;
};

export default function Sidebar({
  isSidebarOpen,
  conversations,
  activeConversationId,
  onCreateConversation,
  onSelectConversation,
  onRenameConversation,
  onLogout,
  onConversationMouseDown,
}: SidebarProps) {
  const [openPopoverId, setOpenPopoverId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);

  const handleRename = async (conversationId: string) => {
    if (renamingId) return;
    setRenamingId(conversationId);
    const ok = await onRenameConversation(conversationId, draftTitle);
    setRenamingId(null);
    if (!ok) return;
    setOpenPopoverId(null);
  };

  return (
    <aside
      className={`${
        isSidebarOpen ? "w-[260px]" : "w-0"
      } overflow-hidden border-r border-slate-200 transition-all duration-300`}
    >
      <div className="flex h-full flex-col p-3">
        <Button
          onClick={onCreateConversation}
          color="default"
          radius="lg"
          className="border-0 data-[hover=true]:border-0 data-[hover=true]:shadow-none"
          variant="light"
        >
          + 新建对话
        </Button>

        <div className="mt-4 flex-1 space-y-2 overflow-y-auto">
          {conversations.map((item) => {
            const isActive = item.id === activeConversationId;
            return (
              <div
                key={item.id}
                onMouseDown={onConversationMouseDown}
                className={`conversation-ripple-container flex w-full items-center rounded-lg border text-sm transition-colors transition-transform duration-150 active:scale-[0.98] ${
                  isActive ? "bg-[#EAEAEA]" : "border-transparent hover:bg-[#EFEFEF]"
                }`}
              >
                <div
                  onClick={() => onSelectConversation(item.id)}
                  className="relative z-[1] flex-1 cursor-pointer bg-transparent px-3 py-2 text-left"
                >
                  {item.title}
                </div>

                <Popover
                  placement="bottom"
                  isOpen={openPopoverId === item.id}
                  onOpenChange={(isOpen) => {
                    setOpenPopoverId(isOpen ? item.id : null);
                    if (isOpen) {
                      setDraftTitle(item.title);
                    }
                  }}
                >
                  <PopoverTrigger>
                    <Button
                      isIconOnly
                      variant="light"
                      disableRipple
                      className="relative z-[1] ml-auto border-0 bg-transparent text-[#838383] shadow-none outline-none ring-0 data-[hover=true]:border-0 data-[hover=true]:border-transparent data-[hover=true]:bg-transparent data-[hover=true]:text-[#2E2E2E] data-[hover=true]:shadow-none data-[pressed=true]:border-0 data-[pressed=true]:shadow-none data-[focus=true]:border-0 data-[focus=true]:outline-none data-[focus=true]:ring-0 data-[focus=true]:shadow-none data-[focus-visible=true]:border-0 data-[focus-visible=true]:outline-none data-[focus-visible=true]:ring-0"
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
                        value={openPopoverId === item.id ? draftTitle : ""}
                        onValueChange={setDraftTitle}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            void handleRename(item.id);
                          }
                        }}
                        classNames={{
                          label: "text-[12px] text-slate-500",
                          input: "text-sm",
                        }}
                      />
                      <div className="flex items-center justify-end gap-2">
                        <Button
                          size="sm"
                          color="primary"
                          className="h-7 min-w-0 px-3 text-xs"
                          isLoading={renamingId === item.id}
                          onPress={() => void handleRename(item.id)}
                        >
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
                  className="relative z-[1] ml-auto border-0 bg-transparent text-[#838383] shadow-none outline-none ring-0 data-[hover=true]:border-0 data-[hover=true]:border-transparent data-[hover=true]:bg-transparent data-[hover=true]:text-[#2E2E2E] data-[hover=true]:shadow-none data-[pressed=true]:border-0 data-[pressed=true]:shadow-none data-[focus=true]:border-0 data-[focus=true]:outline-none data-[focus=true]:ring-0 data-[focus=true]:shadow-none data-[focus-visible=true]:border-0 data-[focus-visible=true]:outline-none data-[focus-visible=true]:ring-0"
                >
                  <Trash />
                </Button>
              </div>
            );
          })}
        </div>

        <Button onClick={onLogout} variant="bordered" className="text-sm text-slate-600 hover:bg-white">
          退出登录
        </Button>
      </div>
    </aside>
  );
}
