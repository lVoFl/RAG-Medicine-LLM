import request from "./request";
import type { New_Conservation, Update_Conservation } from "../types/conservation"
import type { Message } from "../types/conservation"

const GENERATE_TIMEOUT_MS = 300000;

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
    Get_Message(c_id: string){
        return request.get(`/api/conversations/${c_id}/messages`);
    }
}

export default api;
