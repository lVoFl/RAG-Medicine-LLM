import request from "./request";
import type { New_Conservation, Update_Conservation } from "../types/conservation"
import type { Message } from "../types/conservation"

const GENERATE_TIMEOUT_MS = 300000;

type StreamEvent =
    | { type: "delta"; text?: string }
    | { type: "end"; answer?: string; usage?: unknown; params?: unknown }
    | { type: "persisted"; user_message?: unknown; assistant_message?: unknown; usage?: unknown; params?: unknown }
    | { type: "error"; error?: string };

type StreamHandlers = {
    onDelta?: (text: string) => void;
    onDone?: (event: StreamEvent) => void;
    onPersisted?: (event: StreamEvent) => void;
};

const api = {
    Create_Conversation(data: New_Conservation) {
        return request.post('/api/conversations', data);
    },
    Get_Conversation() {
        return request.get('/api/conversations');
    },
    Get_Id_Conversation(id: string) {
        return request.get(`/api/conversations/${id}`);
    },
    Patch_Conservation(data: Update_Conservation, c_id: string){
        return request.patch(`/api/conversations/${c_id}`, data);
    },

    Add_Message(data: Message, c_id: string){
        return request.post(`/api/conversations/${c_id}/messages`, data);
    },
    SendAndGenerate(data: { question: string; context?: string }, c_id: string) {
        return request.post(`/api/model/conversations/${c_id}/generate`, data, {
            timeout: GENERATE_TIMEOUT_MS,
        });
    },
    async SendAndGenerateStream(
        data: { question: string; context?: string },
        c_id: string,
        handlers: StreamHandlers = {}
    ) {
        const token = localStorage.getItem("token");
        const response = await fetch(`http://localhost:3000/api/model/conversations/${c_id}/generate/stream`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...(token ? { Authorization: `Bearer ${token}` } : {}),
            },
            body: JSON.stringify(data),
        });

        if (!response.ok || !response.body) {
            let errorMessage = `请求失败: ${response.status}`;
            try {
                const data = await response.json();
                if (typeof data?.error === "string" && data.error.trim()) {
                    errorMessage = data.error;
                }
            } catch {
                // ignore
            }
            throw new Error(errorMessage);
        }

        const decoder = new TextDecoder();
        const reader = response.body.getReader();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            let boundary = buffer.indexOf("\n\n");
            while (boundary !== -1) {
                const frame = buffer.slice(0, boundary);
                buffer = buffer.slice(boundary + 2);

                const dataLine = frame
                    .split("\n")
                    .map((line) => line.trim())
                    .find((line) => line.startsWith("data:"));
                if (!dataLine) {
                    boundary = buffer.indexOf("\n\n");
                    continue;
                }

                const rawData = dataLine.slice(5).trim();
                if (!rawData) {
                    boundary = buffer.indexOf("\n\n");
                    continue;
                }

                let event: StreamEvent;
                try {
                    event = JSON.parse(rawData);
                } catch {
                    boundary = buffer.indexOf("\n\n");
                    continue;
                }

                if (event.type === "delta") {
                    const text = typeof event.text === "string" ? event.text : "";
                    if (text) handlers.onDelta?.(text);
                } else if (event.type === "end") {
                    handlers.onDone?.(event);
                } else if (event.type === "persisted") {
                    handlers.onPersisted?.(event);
                } else if (event.type === "error") {
                    throw new Error(event.error || "流式生成失败");
                }

                boundary = buffer.indexOf("\n\n");
            }
        }
    },
    Get_Message(c_id: string){
        return request.get(`/api/conversations/${c_id}/messages`);
    }
}

export default api;
