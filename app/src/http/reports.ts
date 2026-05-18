import request from "./request";

const reportsApi = {
  async uploadPdfAndExtractText(file: File, modelVersion = "vlm") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_version", modelVersion);

    const response = await request.post("/api/reports/mineru/upload-pdf/extract-text", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      timeout: 240000,
    });

    return response.data as {
      provider?: string;
      endpoint?: string;
      message?: string;
      batch_id?: string;
      data_id?: string;
      text?: string;
      result_meta?: unknown;
    };
  },
};

export default reportsApi;
