export interface MedicalDocument {
  id: string | number;
  title: string;
  category?: string | null;
  source?: string | null;
  version?: string | null;
  tags?: string[];
  ingest_source?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface MedicalDocumentPayload {
  title: string;
  category?: string;
  source?: string;
  version?: string;
  tags?: string[];
}

export interface UploadTextPayload {
  title: string;
  source: string;
  text: string;
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
