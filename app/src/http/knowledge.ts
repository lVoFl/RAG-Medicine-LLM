import request from "./request";
import type {
  KnowledgeListResponse,
  MedicalDocument,
  MedicalDocumentPayload,
  UploadTextPayload,
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
  getIndexStatus() {
    return request.get<{
      status: string;
      last_reindexed_at?: string | null;
      last_error?: string | null;
      updated_at?: string | null;
    }>("/api/knowledge/index-status");
  },
  reindex() {
    return request.post<{ ok: boolean }>("/api/knowledge/reindex");
  },
  uploadText(data: UploadTextPayload) {
    return request.post<{ ok: boolean; document: MedicalDocument }>("/api/knowledge/upload-text", data, {
      timeout: 600000,
    });
  },
};

export default knowledgeApi;
