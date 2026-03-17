"""
使用 docling 批量处理 origin_file 中的 PDF 文档
并进行语义分段，输出 JSON 供 RAG 检索使用

输出目录结构：
  database/processed_pdf/
    ├── Hypertension/
    │   ├── xxx.json          # 每个 PDF 的分段结果
    │   └── ...
    └── index.json            # 所有文档的汇总索引
"""

import os
import json
import re
import time
import logging
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

# ─── 配置 ────────────────────────────────────────────────────────────────────

ORIGIN_DIR = Path(__file__).parent / "origin_file"
OUTPUT_DIR = Path(__file__).parent / "processed_pdf"

# HybridChunker 参数（以 token 数衡量，中文约 1 token ≈ 1.5 字）
MAX_TOKENS = 512
MERGE_PEERS = True

# 截断文末无关内容的标记词
TAIL_MARKERS = [
    "参考文献",
    "利益冲突",
    "《基层医疗卫生机构合理用药指南》编写专家组",
    "心血管系统疾病合理用药指南编写组",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── 文本后处理 ───────────────────────────────────────────────────────────────

def truncate_tail(text: str) -> str:
    """截断参考文献等文末无关内容"""
    for marker in TAIL_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    return text


def remove_pdf_spaces(text: str) -> str:
    """
    移除 docling 解析中文 PDF 时产生的多余空格。
    常见问题：
      - 数字被空格隔开：  "2 0 2 4"  → "2024"
      - 字母缩写被隔开：  "k P a"    → "kPa"
      - 小数点被空格包围："1 . 6"    → "1.6"
      - 括号内多余空格：  "( 中国 )" → "(中国)"
      - 标点前多余空格：  "率 ,"     → "率,"
      - 引用角标：        "[ 1 ]"    → "[1]"
    """
    # 1. 合并单字符序列 "X Y Z" → "XYZ"
    #    匹配：非字母/数字前 + 单字符 + (空格+单字符)若干 + 非字母/数字后
    text = re.sub(
        r'(?<![a-zA-Z\d])([a-zA-Z\d])(?: ([a-zA-Z\d]))+(?![a-zA-Z\d])',
        lambda m: m.group(0).replace(' ', ''),
        text,
    )

    # 2. 处理 "mmH g" 类残余：大写字母 + 空格 + 单个小写字母
    text = re.sub(r'([A-Z]) ([a-z])(?![a-zA-Z])', r'\1\2', text)

    # 3. 数字小数点 "1 . 6" → "1.6"
    text = re.sub(r'(\d+) \. (\d+)', r'\1.\2', text)

    # 4. 括号内外空格 "( text )" → "(text)"
    text = re.sub(r'\( ', '(', text)
    text = re.sub(r' \)', ')', text)

    # 5. 中文单引号 "' text '" → "'text'"
    text = re.sub(r"' ", "'", text)
    text = re.sub(r" '", "'", text)

    # 6. 标点符号前多余空格
    text = re.sub(r' ([，。、；：！？‰°])', r'\1', text)
    text = re.sub(r' %', '%', text)
    text = re.sub(r' ,', ',', text)

    # 7. 引用角标 "[ 1 ]" → "[1]"
    text = re.sub(
        r'\[ ?(\d+(?:[, ]+\d+)*) ?\]',
        lambda m: '[' + re.sub(r'\s', '', m.group(1)) + ']',
        text,
    )

    # 8. 比较运算符两侧空格 " ≥ " → "≥"
    text = re.sub(r' ([≥≤><≈]) ', r'\1', text)

    # 9. 合并多余连续空格
    text = re.sub(r'  +', ' ', text)

    return text


def clean_chunk_text(text: str) -> str:
    """对单个 chunk 文本做清洗"""
    text = re.sub(r"[··•・·]{1,2}\s*\d+\s*[··•・·]{0,2}", "", text)
    text = re.sub(r"DOI:.*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = remove_pdf_spaces(text)
    return text.strip()


# ─── 核心处理 ─────────────────────────────────────────────────────────────────

def build_converter() -> DocumentConverter:
    """构建 docling PDF 转换器"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False  # 需要新版 transformers 支持 rt_detr_v2，暂时关闭
    # pipeline_options.do_layout = False

    # 降低页面渲染分辨率以避免大页 OOM（std::bad_alloc）
    # 默认值约为 2.0（~144 DPI），降至 1.0（72 DPI）足够版面分析
    pipeline_options.images_scale = 0.8

    # 不需要保存整页截图和图片内容（纯文本 RAG 场景）
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def build_chunker() -> HybridChunker:
    """
    构建 HybridChunker。
    不指定 tokenizer 时 docling 使用内置字符估算，无需下载额外模型。
    """
    try:
        # 优先尝试多语言 tokenizer（需联网下载一次）
        chunker = HybridChunker(
            tokenizer="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_tokens=MAX_TOKENS,
            merge_peers=MERGE_PEERS,
        )
        log.info("HybridChunker: 使用 multilingual tokenizer")
        return chunker
    except Exception as e:
        log.warning(f"加载 multilingual tokenizer 失败 ({e})，使用默认模式")
        return HybridChunker(max_tokens=MAX_TOKENS, merge_peers=MERGE_PEERS)


def extract_chunk_text(chunk) -> str:
    """
    兼容不同 docling 版本取 chunk 文本。
    新版：chunk.text
    旧版：chunk.text 或通过 chunk.meta.doc_items 拼接
    """
    # 优先取 .text 属性
    if hasattr(chunk, "text") and chunk.text:
        return chunk.text

    # 兼容：通过 doc_items 拼接
    if hasattr(chunk, "meta") and chunk.meta and hasattr(chunk.meta, "doc_items"):
        parts = []
        for item in chunk.meta.doc_items:
            if hasattr(item, "text") and item.text:
                parts.append(item.text)
        if parts:
            return "\n".join(parts)

    return ""


def extract_headings(chunk) -> str:
    """从 chunk.meta 提取层级标题路径"""
    if not (hasattr(chunk, "meta") and chunk.meta):
        return ""
    meta = chunk.meta
    if hasattr(meta, "headings") and meta.headings:
        return " > ".join(str(h) for h in meta.headings if h)
    return ""


def extract_page_range(chunk) -> str:
    """从 chunk.meta.doc_items 提取页码范围"""
    if not (hasattr(chunk, "meta") and chunk.meta):
        return ""
    if not hasattr(chunk.meta, "doc_items"):
        return ""
    pages = set()
    for item in chunk.meta.doc_items:
        prov_list = getattr(item, "prov", None) or []
        for prov in prov_list:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                pages.add(page_no)
    if not pages:
        return ""
    return f"p{min(pages)}" if len(pages) == 1 else f"p{min(pages)}-p{max(pages)}"


def process_pdf(pdf_path: Path, converter: DocumentConverter, chunker: HybridChunker) -> list[dict]:
    """处理单个 PDF，返回语义分段列表"""
    log.info(f"  转换: {pdf_path.name}")
    try:
        result = converter.convert(str(pdf_path))
    except Exception as e:
        log.error(f"  转换失败: {e}")
        return []

    doc = result.document

    # 诊断：检查文档是否有内容
    try:
        md_preview = doc.export_to_markdown()
        md_len = len(md_preview)
        log.info(f"  Markdown 长度: {md_len} 字符")
        if md_len < 100:
            log.warning(f"  文档内容极少，可能提取失败。预览: {md_preview[:200]!r}")
    except Exception as e:
        log.warning(f"  export_to_markdown 失败: {e}")
        md_len = 0

    if md_len == 0:
        log.warning(f"  文档无内容，跳过")
        return []

    # 语义分段
    try:
        chunks_raw = list(chunker.chunk(doc))
    except Exception as e:
        log.error(f"  分块失败: {e}")
        return []

    log.info(f"  原始 chunk 数: {len(chunks_raw)}")

    category = pdf_path.parent.name
    source = pdf_path.name
    chunks_out = []

    for i, chunk in enumerate(chunks_raw):
        text = clean_chunk_text(extract_chunk_text(chunk))
        if not text:
            continue

        chunks_out.append({
            "chunk_id":   i,
            "source":     source,
            "category":   category,
            "headings":   remove_pdf_spaces(extract_headings(chunk)),
            "content":    text,
            "page_range": extract_page_range(chunk),
        })

    log.info(f"  有效 chunk 数: {len(chunks_out)}")
    return chunks_out


# ─── 批量处理 ─────────────────────────────────────────────────────────────────

def batch_process(force: bool = False):
    """
    force=True 时强制重新处理已有结果的 PDF
    """
    if not ORIGIN_DIR.exists():
        log.error(f"原始文件目录不存在: {ORIGIN_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    converter = build_converter()
    chunker = build_chunker()

    pdf_files = sorted(ORIGIN_DIR.rglob("*.pdf"))
    log.info(f"发现 {len(pdf_files)} 个 PDF 文件\n")

    all_index = []

    for pdf_path in pdf_files:
        rel = pdf_path.relative_to(ORIGIN_DIR)
        out_dir = OUTPUT_DIR / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (pdf_path.stem + ".json")

        if out_path.exists() and not force:
            log.info(f"[跳过] 已处理: {rel}")
            with open(out_path, encoding="utf-8") as f:
                chunks = json.load(f)
            all_index.extend(chunks)
            continue

        log.info(f"{'─'*60}")
        log.info(f"处理: {rel}")
        t0 = time.time()

        chunks = process_pdf(pdf_path, converter, chunker)

        if chunks:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            log.info(f"  保存 -> {out_path}  ({time.time()-t0:.1f}s)")
            all_index.extend(chunks)
        else:
            log.warning(f"  无输出，跳过保存  ({time.time()-t0:.1f}s)")

    index_path = OUTPUT_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_index, f, ensure_ascii=False, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"全部完成: {len(pdf_files)} 个 PDF -> {len(all_index)} 个 chunk")
    log.info(f"汇总索引 -> {index_path}")


# ─── 单文件诊断入口 ───────────────────────────────────────────────────────────

def diagnose(pdf_path: str):
    """对单个 PDF 进行诊断，打印前 3 个 chunk 内容"""
    path = Path(pdf_path)
    converter = build_converter()
    chunker = build_chunker()

    result = converter.convert(str(path))
    doc = result.document

    md = doc.export_to_markdown()
    print(f"\n=== Markdown 前 500 字 ===\n{md[:500]}\n")

    chunks = list(chunker.chunk(doc))
    print(f"共 {len(chunks)} 个 chunk\n")
    for i, c in enumerate(chunks[:3]):
        text = extract_chunk_text(c)
        headings = extract_headings(c)
        print(f"[chunk {i}] headings={headings!r}")
        print(text[:300])
        print("---")


# ─── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # python batch_pdf.py <pdf_path>  -> 诊断模式
        diagnose(sys.argv[1])
    else:
        # 正常批量处理（force=True 重新处理所有）
        batch_process(force=True)
