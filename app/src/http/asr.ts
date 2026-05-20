import request from "./request";

const asrApi = {
  async transcribeAudio(file: File, languageHints = "zh,en") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("language_hints", languageHints);

    const response = await request.post("/api/asr/transcribe", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      timeout: 240000,
    });

    return response.data as {
      text?: string;
      task_id?: string;
      model?: string;
      provider?: string;
      raw?: unknown;
    };
  },
};

export default asrApi;
