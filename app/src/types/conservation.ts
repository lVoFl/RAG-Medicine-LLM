export interface New_Conservation {
    title: string;
}

export interface Update_Conservation {
    title: string;
    last_message: JSON;
}

type ChatRole = "user" | "assistant" | "system" | "summary" | "tool";
export interface Content {
    text: string;
    attachments: []
}
export interface Message {
    role: ChatRole;
    content: Content;
}