import type { Message } from "./conservation";

export type Conversation = {
  id: string;
  title: string;
  updatedAt: number;
  messageCount: number;
};

export type ConversationMessage = Message & {
  id?: string | number;
  [key: string]: unknown;
};
