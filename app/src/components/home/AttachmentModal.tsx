import type { ChangeEvent, FormEvent } from "react";
import { Button, Input, Modal, ModalBody, ModalContent, ModalFooter, ModalHeader, Textarea } from "@heroui/react";

type HealthForm = {
  age: string;
  height: string;
  weight: string;
  systolic: string;
  diastolic: string;
  fastingGlucose: string;
  postprandialGlucose: string;
  hba1c: string;
  totalCholesterol: string;
  triglyceride: string;
  ldl: string;
  hdl: string;
};

type AttachmentModalProps = {
  isOpen: boolean;
  healthForm: HealthForm;
  selectedPdfName: string;
  selectedPdfFile: File | null;
  isParsingPdf: boolean;
  parsedReportText: string;
  parseError: string;
  onOpenChange: (open: boolean) => void;
  onHealthFieldChange: (key: keyof HealthForm, value: string) => void;
  onPdfChange: (e: ChangeEvent<HTMLInputElement>) => void;
  onParsePdf: () => Promise<void>;
  onRemovePdf: () => void;
  onParsedTextChange: (value: string) => void;
  onSubmit: (e: FormEvent<HTMLFormElement>) => void;
};

export default function AttachmentModal({
  isOpen,
  healthForm,
  selectedPdfName,
  selectedPdfFile,
  isParsingPdf,
  parsedReportText,
  parseError,
  onOpenChange,
  onHealthFieldChange,
  onPdfChange,
  onParsePdf,
  onRemovePdf,
  onParsedTextChange,
  onSubmit,
}: AttachmentModalProps) {
  return (
    <Modal isOpen={isOpen} onOpenChange={onOpenChange} size="2xl" placement="center">
      <ModalContent>
        <form onSubmit={onSubmit}>
          <ModalHeader>补充身体指标与检测报告</ModalHeader>
          <ModalBody>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <Input label="年龄" type="number" value={healthForm.age} onValueChange={(v) => onHealthFieldChange("age", v)} />
              <Input label="身高(cm)" type="number" value={healthForm.height} onValueChange={(v) => onHealthFieldChange("height", v)} />
              <Input label="体重(kg)" type="number" value={healthForm.weight} onValueChange={(v) => onHealthFieldChange("weight", v)} />
              <Input label="收缩压(mmHg)" type="number" value={healthForm.systolic} onValueChange={(v) => onHealthFieldChange("systolic", v)} />
              <Input label="舒张压(mmHg)" type="number" value={healthForm.diastolic} onValueChange={(v) => onHealthFieldChange("diastolic", v)} />
              <Input label="空腹血糖(mmol/L)" type="number" value={healthForm.fastingGlucose} onValueChange={(v) => onHealthFieldChange("fastingGlucose", v)} />
              <Input label="餐后血糖(mmol/L)" type="number" value={healthForm.postprandialGlucose} onValueChange={(v) => onHealthFieldChange("postprandialGlucose", v)} />
              <Input label="糖化血红蛋白(%)" type="number" value={healthForm.hba1c} onValueChange={(v) => onHealthFieldChange("hba1c", v)} />
              <Input label="总胆固醇(mmol/L)" type="number" value={healthForm.totalCholesterol} onValueChange={(v) => onHealthFieldChange("totalCholesterol", v)} />
              <Input label="甘油三酯(mmol/L)" type="number" value={healthForm.triglyceride} onValueChange={(v) => onHealthFieldChange("triglyceride", v)} />
              <Input label="LDL-C(mmol/L)" type="number" value={healthForm.ldl} onValueChange={(v) => onHealthFieldChange("ldl", v)} />
              <Input label="HDL-C(mmol/L)" type="number" value={healthForm.hdl} onValueChange={(v) => onHealthFieldChange("hdl", v)} />
            </div>

            <div className="mt-2">
              <label className="mb-1 block text-sm text-slate-600">上传检测报告（PDF）</label>
              <div className="flex flex-wrap items-center gap-2">
                <input
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={onPdfChange}
                  className="block flex-1 text-sm text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-2 file:text-sm file:text-slate-700"
                />
                <Button
                  type="button"
                  variant="bordered"
                  onPress={() => void onParsePdf()}
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
                    onClick={onRemovePdf}
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
                  onValueChange={onParsedTextChange}
                  placeholder="解析结果会显示在这里"
                  variant="bordered"
                />
              </div>
            ) : null}
          </ModalBody>
          <ModalFooter>
            <Button variant="light" onPress={() => onOpenChange(false)}>
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
  );
}
