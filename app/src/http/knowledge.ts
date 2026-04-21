import request from "./request";
import type {
  KnowledgeListResponse,
  MedicalDocument,
  MedicalDocumentPayload,
} from "../types/knowledge";

const knowledgeApi = {
  list(params?: { keyword?: string; page?: number; pageSize?: number }) {
    return request.get<KnowledgeListResponse>("/api/knowledge/documents", { params });
  },
  detail(id: string | number) {
    return request.get<MedicalDocument>(`/api/knowledge/documents/${id}`);
  },
  create(data: MedicalDocumentPayload) {
    return request.post<MedicalDocument>("/api/knowledge/documents", data);
  },
  update(id: string | number, data: Partial<MedicalDocumentPayload>) {
    return request.patch<MedicalDocument>(`/api/knowledge/documents/${id}`, data);
  },
  remove(id: string | number) {
    return request.delete(`/api/knowledge/documents/${id}`);
  },
};

export default knowledgeApi;
