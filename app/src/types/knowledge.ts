export interface MedicalDocument {
  id: string | number;
  title: string;
  category?: string | null;
  summary?: string | null;
  content: string;
  source?: string | null;
  version?: string | null;
  tags?: string[];
  created_at?: string;
  updated_at?: string;
}

export interface MedicalDocumentPayload {
  title: string;
  category?: string;
  summary?: string;
  content: string;
  source?: string;
  version?: string;
  tags?: string[];
}

export interface KnowledgeListResponse {
  list: MedicalDocument[];
  pagination: {
    page: number;
    pageSize: number;
    total: number;
  };
}
