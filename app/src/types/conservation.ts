export interface New_Conservation {
    title: string;
}

export interface Update_Conservation {
    title?: string;
    last_message?: unknown;
}

type ChatRole = "user" | "assistant" | "system" | "summary" | "tool";

export interface RagDoc {
    chunk_id?: string | number;
    source?: string;
    headings?: string;
    content?: string;
    score?: number;
    [key: string]: unknown;
}

export interface Content {
    text: string;
    attachments: unknown[];
    retrieved_docs?: RagDoc[];
    usage?: unknown;
    params?: unknown;
    [key: string]: unknown;
}
export interface Message {
    role: ChatRole;
    content: Content;
    [key: string]: unknown;
}
