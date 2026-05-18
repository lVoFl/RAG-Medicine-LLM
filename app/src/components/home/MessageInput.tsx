import { useEffect, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import { Button, Input, Modal, ModalBody, ModalContent, ModalFooter, ModalHeader, Textarea } from "@heroui/react";
import reportsApi from "../../http/reports";

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
            className="h-9 min-h-9 w-9 min-w-9 rounded-full p-0 text-slate-700"
            title="语音输入功能开发中"
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

      <Modal isOpen={isAttachmentModalOpen} onOpenChange={setIsAttachmentModalOpen} size="2xl" placement="center">
        <ModalContent>
          <form onSubmit={handleAttachmentSubmit}>
            <ModalHeader>补充身体指标与检测报告</ModalHeader>
            <ModalBody>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <Input label="年龄" type="number" value={healthForm.age} onValueChange={(v) => handleHealthFieldChange("age", v)} />
                <Input label="身高(cm)" type="number" value={healthForm.height} onValueChange={(v) => handleHealthFieldChange("height", v)} />
                <Input label="体重(kg)" type="number" value={healthForm.weight} onValueChange={(v) => handleHealthFieldChange("weight", v)} />
                <Input label="收缩压(mmHg)" type="number" value={healthForm.systolic} onValueChange={(v) => handleHealthFieldChange("systolic", v)} />
                <Input label="舒张压(mmHg)" type="number" value={healthForm.diastolic} onValueChange={(v) => handleHealthFieldChange("diastolic", v)} />
                <Input label="空腹血糖(mmol/L)" type="number" value={healthForm.fastingGlucose} onValueChange={(v) => handleHealthFieldChange("fastingGlucose", v)} />
                <Input label="餐后血糖(mmol/L)" type="number" value={healthForm.postprandialGlucose} onValueChange={(v) => handleHealthFieldChange("postprandialGlucose", v)} />
                <Input label="糖化血红蛋白(%)" type="number" value={healthForm.hba1c} onValueChange={(v) => handleHealthFieldChange("hba1c", v)} />
                <Input label="总胆固醇(mmol/L)" type="number" value={healthForm.totalCholesterol} onValueChange={(v) => handleHealthFieldChange("totalCholesterol", v)} />
                <Input label="甘油三酯(mmol/L)" type="number" value={healthForm.triglyceride} onValueChange={(v) => handleHealthFieldChange("triglyceride", v)} />
                <Input label="LDL-C(mmol/L)" type="number" value={healthForm.ldl} onValueChange={(v) => handleHealthFieldChange("ldl", v)} />
                <Input label="HDL-C(mmol/L)" type="number" value={healthForm.hdl} onValueChange={(v) => handleHealthFieldChange("hdl", v)} />
              </div>

              <div className="mt-2">
                <label className="mb-1 block text-sm text-slate-600">上传检测报告（PDF）</label>
                <div className="flex flex-wrap items-center gap-2">
                  <input
                    type="file"
                    accept="application/pdf,.pdf"
                    onChange={handlePdfChange}
                    className="block flex-1 text-sm text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-2 file:text-sm file:text-slate-700"
                  />
                  <Button
                    type="button"
                    variant="bordered"
                    onPress={() => void handleParsePdf()}
                    isLoading={isParsingPdf}
                    isDisabled={!selectedPdfFile}
                  >
                    解析PDF
                  </Button>
                </div>
                {selectedPdfName ? (
                  <div className="mt-2 inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-700">
                    <span className="max-w-[360px] truncate">{selectedPdfName}</span>
                    <button
                      type="button"
                      onClick={handleRemovePdf}
                      className="inline-flex h-4 w-4 items-center justify-center rounded-full text-slate-500 hover:bg-slate-200 hover:text-slate-700"
                      aria-label="删除已上传PDF"
                      title="删除已上传PDF"
                    >
                      ×
                    </button>
                  </div>
                ) : null}
              </div>

              {parseError ? <p className="text-xs text-red-500">{parseError}</p> : null}

              {parsedReportText ? (
                <div>
                  <label className="mb-1 block text-sm text-slate-600">解析结果（Markdown，可编辑）</label>
                  <Textarea
                    minRows={8}
                    value={parsedReportText}
                    onValueChange={setParsedReportText}
                    placeholder="解析结果会显示在这里"
                    variant="bordered"
                  />
                </div>
              ) : null}
            </ModalBody>
            <ModalFooter>
              <Button variant="light" onPress={() => setIsAttachmentModalOpen(false)}>
                取消
              </Button>
              <Button
                type="submit"
                color="primary"
                isDisabled={Boolean(selectedPdfFile) && !parsedReportText.trim()}
              >
                保存
              </Button>
            </ModalFooter>
          </form>
        </ModalContent>
      </Modal>
    </section>
  );
}
