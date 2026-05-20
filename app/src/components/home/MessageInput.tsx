import { Suspense, lazy, useEffect, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import { Button, Textarea } from "@heroui/react";
import reportsApi from "../../http/reports";
import asrApi from "../../http/asr";
const AttachmentModal = lazy(() => import("./AttachmentModal"));

type MessageInputProps = {
  inputValue: string;
  isSending: boolean;
  onInputChange: (value: string) => void;
  onSubmit: (e?: FormEvent) => void;
  onSaveSupplementalData?: (data: { healthProfile: Record<string, string>; reportText: string }) => void;
  activeConversationId?: string;
};

export default function MessageInput({
  inputValue,
  isSending,
  onInputChange,
  onSubmit,
  onSaveSupplementalData,
  activeConversationId,
}: MessageInputProps) {
  const [isAttachmentModalOpen, setIsAttachmentModalOpen] = useState(false);
  const [selectedPdfName, setSelectedPdfName] = useState("");
  const [selectedPdfFile, setSelectedPdfFile] = useState<File | null>(null);
  const [isParsingPdf, setIsParsingPdf] = useState(false);
  const [parsedReportText, setParsedReportText] = useState("");
  const [parseError, setParseError] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribingAudio, setIsTranscribingAudio] = useState(false);
  const [audioError, setAudioError] = useState("");
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [healthForm, setHealthForm] = useState({
    age: "",
    height: "",
    weight: "",
    systolic: "",
    diastolic: "",
    fastingGlucose: "",
    postprandialGlucose: "",
    hba1c: "",
    totalCholesterol: "",
    triglyceride: "",
    ldl: "",
    hdl: "",
  });

  useEffect(() => {
    setIsAttachmentModalOpen(false);
    setSelectedPdfName("");
    setSelectedPdfFile(null);
    setIsParsingPdf(false);
    setParsedReportText("");
    setParseError("");
    setAudioError("");
    setHealthForm({
      age: "",
      height: "",
      weight: "",
      systolic: "",
      diastolic: "",
      fastingGlucose: "",
      postprandialGlucose: "",
      hba1c: "",
      totalCholesterol: "",
      triglyceride: "",
      ldl: "",
      hdl: "",
    });
  }, [activeConversationId]);

  const handleHealthFieldChange = (key: keyof typeof healthForm, value: string) => {
    setHealthForm((prev) => ({ ...prev, [key]: value }));
  };

  const handlePdfChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      setSelectedPdfName("");
      setSelectedPdfFile(null);
      setParsedReportText("");
      return;
    }
    setSelectedPdfName(file.name);
    setSelectedPdfFile(file);
    setParsedReportText("");
    setParseError("");
  };

  const handleRemovePdf = () => {
    setSelectedPdfName("");
    setSelectedPdfFile(null);
    setParsedReportText("");
    setParseError("");
  };

  const handleParsePdf = async () => {
    if (!selectedPdfFile) {
      setParseError("请先选择一个 PDF 文件");
      return;
    }

    try {
      setIsParsingPdf(true);
      setParseError("");
      const result = await reportsApi.uploadPdfAndExtractText(selectedPdfFile, "vlm");
      setParsedReportText(String(result?.text ?? ""));
    } catch (error) {
      const message =
        typeof error === "object" &&
        error !== null &&
        "response" in error &&
        typeof (error as { response?: { data?: { error?: string } } }).response?.data?.error === "string"
          ? (error as { response: { data: { error: string } } }).response.data.error
          : "PDF 解析失败，请稍后重试";
      setParseError(message);
    } finally {
      setIsParsingPdf(false);
    }
  };

  const handleAttachmentSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    onSaveSupplementalData?.({
      healthProfile: healthForm,
      reportText: parsedReportText.trim(),
    });
    setIsAttachmentModalOpen(false);
  };

  const handleStartRecording = async () => {
    try {
      setAudioError("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const recordedChunks: Blob[] = [];

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunks.push(event.data);
        }
      };

      recorder.onstop = async () => {
        try {
          setIsTranscribingAudio(true);
          const mimeType = recorder.mimeType || "audio/webm";
          const blob = new Blob(recordedChunks, { type: mimeType });
          if (!blob.size) {
            setAudioError("录音内容为空，请重试");
            return;
          }
          const file = new File([blob], `voice-${Date.now()}.webm`, { type: mimeType });
          const result = await asrApi.transcribeAudio(file, "zh,en");
          const text = String(result?.text || "").trim();
          if (!text) {
            setAudioError("语音识别未返回文本，请重试");
            return;
          }
          onInputChange(inputValue ? `${inputValue}\n${text}` : text);
        } catch (error) {
          const message =
            typeof error === "object" &&
            error !== null &&
            "response" in error &&
            typeof (error as { response?: { data?: { error?: string } } }).response?.data?.error === "string"
              ? (error as { response: { data: { error: string } } }).response.data.error
              : "语音转文字失败，请稍后重试";
          setAudioError(message);
        } finally {
          setIsTranscribingAudio(false);
          stream.getTracks().forEach((track) => track.stop());
          setMediaRecorder(null);
          setAudioChunks([]);
        }
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch {
      setAudioError("无法访问麦克风，请检查浏览器权限");
    }
  };

  const handleStopRecording = () => {
    if (!mediaRecorder || mediaRecorder.state !== "recording") return;
    mediaRecorder.stop();
    setIsRecording(false);
  };

  return (
    <section className="border-t border-slate-200 bg-white px-4 py-4 md:px-6">
      <form onSubmit={onSubmit} className="mx-auto w-full max-w-4xl">
        <div className="flex items-center gap-2 rounded-full border border-slate-300 bg-white px-3 py-1.5 shadow-sm">
          <Button
            type="button"
            isIconOnly
            variant="light"
            className="flex h-9 min-h-9 w-9 min-w-9 items-center justify-center rounded-full p-0 text-slate-700"
            title="补充身体指标与检测报告"
            onPress={() => setIsAttachmentModalOpen(true)}
          >
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M12 5v14" />
              <path d="M5 12h14" />
            </svg>
          </Button>

          <Textarea
            minRows={1}
            value={inputValue}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSubmit();
              }
            }}
            placeholder="请输入你的问题"
            variant="flat"
            classNames={{
              base: "flex-1",
              inputWrapper:
                "bg-transparent shadow-none border-0 py-0 data-[hover=true]:bg-transparent group-data-[focus=true]:bg-transparent",
              innerWrapper: "items-center",
              input: "min-h-[24px] max-h-32 py-0.5 text-base leading-6 text-slate-700 placeholder:text-slate-400",
            }}
          />

          <Button
            type="button"
            isIconOnly
            variant="light"
            className={`h-9 min-h-9 w-9 min-w-9 rounded-full p-0 ${isRecording ? "text-red-500" : "text-slate-700"}`}
            title={isRecording ? "停止录音" : "开始录音"}
            isDisabled={isTranscribingAudio}
            isLoading={isTranscribingAudio}
            onPress={() => {
              if (isRecording) {
                handleStopRecording();
                return;
              }
              void handleStartRecording();
            }}
          >
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 1 0 6 0V6a3 3 0 0 0-3-3z" />
              <path d="M19 11a7 7 0 0 1-14 0" />
              <path d="M12 18v3" />
              <path d="M8 21h8" />
            </svg>
          </Button>

          <Button
            type="submit"
            isIconOnly
            disabled={isSending || !inputValue.trim()}
            isDisabled={isSending || !inputValue.trim()}
            className="h-9 min-h-9 w-9 min-w-9 rounded-full bg-black p-0 text-white data-[hover=true]:bg-slate-800"
            title="发送"
          >
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="m5 12 4 4 10-10" />
            </svg>
          </Button>
        </div>
      </form>
      {audioError ? <p className="mx-auto mt-2 w-full max-w-4xl text-xs text-red-500">{audioError}</p> : null}

      {isAttachmentModalOpen ? (
        <Suspense fallback={null}>
          <AttachmentModal
            isOpen={isAttachmentModalOpen}
            healthForm={healthForm}
            selectedPdfName={selectedPdfName}
            selectedPdfFile={selectedPdfFile}
            isParsingPdf={isParsingPdf}
            parsedReportText={parsedReportText}
            parseError={parseError}
            onOpenChange={setIsAttachmentModalOpen}
            onHealthFieldChange={handleHealthFieldChange}
            onPdfChange={handlePdfChange}
            onParsePdf={handleParsePdf}
            onRemovePdf={handleRemovePdf}
            onParsedTextChange={setParsedReportText}
            onSubmit={handleAttachmentSubmit}
          />
        </Suspense>
      ) : null}
    </section>
  );
}
