import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { Button, Input, Textarea } from "@heroui/react";
import knowledgeApi from "../http/knowledge";
import type { MedicalDocument } from "../types/knowledge";

function parseTags(input: string) {
  return input
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function isAdminFromToken() {
  const token = localStorage.getItem("token");
  if (!token) return false;
  const jwt = token.startsWith("Bearer ") ? token.slice(7) : token;
  const parts = jwt.split(".");
  if (parts.length < 2) return false;
  try {
    const payload = JSON.parse(atob(parts[1]));
    return Boolean(payload?.isAdmin);
  } catch {
    return false;
  }
}

export default function KnowledgeManagePage() {
  const navigate = useNavigate();
  const [documents, setDocuments] = useState<MedicalDocument[]>([]);
  const [selectedId, setSelectedId] = useState<string | number | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [indexStatus, setIndexStatus] = useState<{
    status: string;
    last_reindexed_at?: string | null;
    last_error?: string | null;
  } | null>(null);

  const [source, setSource] = useState("");
  const [title, setTitle] = useState("");
  const [tagsInput, setTagsInput] = useState("");
  const [text, setText] = useState("");

  const loadDocuments = async () => {
    setLoading(true);
    try {
      const { data } = await knowledgeApi.list({ page: 1, pageSize: 30 });
      const list = Array.isArray(data.list) ? data.list : [];
      setDocuments(list);
      if (!selectedId) {
        if (list.length) {
          setSelectedId(list[0].id);
          setIsCreating(false);
        } else {
          setIsCreating(true);
        }
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  const loadIndexStatus = async () => {
    try {
      const { data } = await knowledgeApi.getIndexStatus();
      setIndexStatus(data);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    if (!isAdminFromToken()) {
      navigate("/");
      return;
    }
    void loadDocuments();
    void loadIndexStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleUploadText = async () => {
    if (!isCreating) {
      setError("请选择“新建”后再上传文本");
      return;
    }
    if (!title.trim() || !source.trim() || !text.trim()) {
      setError("请填写标题、来源和文本内容");
      return;
    }
    setUploading(true);
    setError("");
    try {
      await knowledgeApi.uploadText({
        title: title.trim(),
        source: source.trim(),
        text: text.trim(),
        tags: parseTags(tagsInput),
      });
      setText("");
      await loadDocuments();
      await loadIndexStatus();
      setIsCreating(false);
    } catch (err) {
      const message =
        (err as { response?: { data?: { error?: string } } })?.response?.data?.error ||
        "上传失败，请稍后重试";
      setError(message);
      await loadIndexStatus();
    } finally {
      setUploading(false);
    }
  };

  const handleSelectDocument = (doc: MedicalDocument) => {
    setSelectedId(doc.id);
    setIsCreating(false);
    setSource(doc.source || "");
    setTitle(doc.title || "");
    setTagsInput(Array.isArray(doc.tags) ? doc.tags.join(", ") : "");
    setText("");
  };

  const handleCreateNew = () => {
    setSelectedId(null);
    setIsCreating(true);
    setTitle("");
    setSource("");
    setTagsInput("");
    setText("");
  };

  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-800">
      <aside className="w-[320px] border-r border-slate-200 bg-white p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="text-lg font-semibold">知识文档列表</div>
          <Button size="sm" color="primary" variant="flat" onPress={handleCreateNew}>
            新建
          </Button>
        </div>
        <p className="mb-4 text-xs text-slate-500">以下为已登记到数据库的文档元信息（文档级）。</p>
        <div className="space-y-2 overflow-y-auto">
          {documents.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => handleSelectDocument(item)}
              className={`w-full rounded-lg border p-3 text-left text-sm transition ${
                String(item.id) === String(selectedId)
                  ? "border-cyan-500 bg-cyan-50"
                  : "border-slate-200 bg-white hover:bg-slate-50"
              }`}
            >
              <div className="line-clamp-1 font-medium">{item.title}</div>
              <div className="mt-1 text-xs text-slate-500">{item.category || "未分类"}</div>
              <div className="mt-1 text-xs text-slate-400">{item.source || "-"}</div>
            </button>
          ))}
          {!documents.length && !loading ? (
            <div className="rounded-lg border border-dashed border-slate-300 p-3 text-xs text-slate-500">
              暂无文档
            </div>
          ) : null}
        </div>
      </aside>

      <main className="flex-1 p-6">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">{isCreating ? "新建并上传到RAG" : "查看文档元信息"}</h1>
            <p className="text-sm text-slate-500">
              {"单一流程：文本分段、embedding、追加 FAISS、热重载。"}
            </p>
          </div>
          <Button variant="bordered" onPress={() => navigate("/")}>
            返回聊天
          </Button>
        </div>

        {indexStatus ? (
          <div className="mb-4 rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-600">
            <p>索引状态：{indexStatus.status}</p>
            <p>最近同步：{indexStatus.last_reindexed_at || "暂无"}</p>
            {indexStatus.last_error ? <p className="text-red-600">最近错误：{indexStatus.last_error}</p> : null}
          </div>
        ) : null}

        {error ? <p className="mb-4 text-sm text-red-600">{error}</p> : null}

        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Input label="标题" value={title} onValueChange={setTitle} isRequired />
            <Input label="文档来源（source）" value={source} onValueChange={setSource} isRequired />
            <Input
              label="标签（逗号分隔）"
              placeholder="如：高血压, 指南, 药物"
              value={tagsInput}
              onValueChange={setTagsInput}
            />
          </div>

          {isCreating ? (
            <>
              <div className="mt-4">
                <Textarea
                  label="文本内容"
                  minRows={18}
                  value={text}
                  onValueChange={setText}
                  isRequired
                />
              </div>
              <div className="mt-4">
                <Button color="primary" onPress={() => void handleUploadText()} isLoading={uploading}>
                  上传文本并入RAG
                </Button>
              </div>
            </>
          ) : (
            <p className="mt-4 text-sm text-slate-500">当前为已有文档。点击左上角“新建”可录入文本并追加到RAG。</p>
          )}
        </div>
      </main>
    </div>
  );
}
