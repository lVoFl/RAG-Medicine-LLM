"""
批量处理 origin_file 中的 Markdown 文档，进行语义分段，
输出 JSON 供 RAG 检索使用。

输出目录结构：
  database/processed_md/
    ├── Diabetes/
    │   ├── xxx.json
    │   └── ...
    └── index.json
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

# ─── 配置 ────────────────────────────────────────────────────────────────────

ORIGIN_DIR = Path(__file__).parent / "origin_file"
OUTPUT_DIR = Path(__file__).parent / "processed_md"

# 字符数上限（中文约 1.5 字/token，512 tokens ≈ 768 字）
MAX_CHARS = 768
# 相邻 chunk 重叠字符数（建议 10%~20%）
OVERLAP_CHARS = 120

# 截断文末无关内容的标记词
TAIL_MARKERS = ["参考文献", "利益冲突"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── 文本清洗 ─────────────────────────────────────────────────────────────────

def truncate_tail(text: str) -> str:
    """截断参考文献等文末无关内容"""
    for marker in TAIL_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    return text


def clean_paragraph(text: str) -> str:
    """清洗段落文本，保留 markdown 的基础可读性"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"DOI:.*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """取文本尾部 overlap 窗口，作为下一 chunk 的前缀"""
    if overlap_chars <= 0:
        return ""
    return text[-overlap_chars:].strip()


def _hard_split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """兜底硬切分：确保每个片段不超过 max_chars"""
    if len(text) <= max_chars:
        return [text]
    if overlap_chars >= max_chars:
        overlap_chars = max_chars // 4
    step = max_chars - max(overlap_chars, 0)
    out = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start += max(step, 1)
    return out


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """按句子边界拆分超长文本，并在相邻 chunk 间添加重叠"""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[。！？.!?])", text)
    chunks, buf = [], ""
    for sent in sentences:
        if len(buf) + len(sent) > max_chars and buf:
            prev = buf.strip()
            chunks.append(prev)
            carry = _overlap_tail(prev, overlap_chars)
            buf = (carry + sent).strip() if carry else sent
            if len(buf) > max_chars:
                # 极端情况下（单句过长）退化为无 overlap
                buf = sent
        else:
            buf += sent
    if buf.strip():
        chunks.append(buf.strip())
    chunks = chunks or [text]

    # 兜底：若仍有超长片段（常见于超长单句），进行硬切分
    fixed = []
    for c in chunks:
        if len(c) > max_chars:
            fixed.extend(_hard_split_with_overlap(c, max_chars, overlap_chars))
        else:
            fixed.append(c)
    return fixed


def _pctl(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = int((len(sorted_vals) - 1) * q)
    return sorted_vals[idx]


def log_chunk_stats(contents: list[str]) -> None:
    """输出 chunk 长度分布，辅助判断 MAX_CHARS 是否合适"""
    if not contents:
        return
    lens = sorted(len(c) for c in contents)
    p50 = _pctl(lens, 0.50)
    p90 = _pctl(lens, 0.90)
    p95 = _pctl(lens, 0.95)
    mx = lens[-1]
    log.info(
        f"  chunk长度(字符) p50/p90/p95/max = {p50}/{p90}/{p95}/{mx} "
        f"(max={MAX_CHARS}, overlap={OVERLAP_CHARS})"
    )
    if p95 < int(MAX_CHARS * 0.45):
        log.warning("  chunk 偏小：可考虑增大 MAX_CHARS（如 896/1024）")
    if mx > MAX_CHARS:
        log.warning("  存在超过 MAX_CHARS 的 chunk，建议检查分句规则")


# ─── Markdown 解析 ───────────────────────────────────────────────────────────

ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
SETEXT_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")


def _normalize_heading_text(text: str) -> str:
    # 去掉 ATX 标题尾部可选的 ### closing markers
    text = re.sub(r"\s+#+\s*$", "", text).strip()
    return text


def parse_md_blocks(text: str) -> list[dict]:
    """
    将 markdown 解析为 block 列表：
      {"type": "heading"|"paragraph", "level": int, "text": str}

    规则：
      1. 识别 ATX 标题（# ~ ######）
      2. 识别 setext 标题（下一行是 === 或 ---）
      3. 代码块按普通段落整体保留
      4. 其余按空行切分段落
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[dict] = []

    in_code = False
    code_fence = ""
    para_buf: list[str] = []

    def flush_para() -> None:
        nonlocal para_buf
        if not para_buf:
            return
        paragraph = clean_paragraph("\n".join(para_buf))
        if paragraph:
            blocks.append({"type": "paragraph", "level": 0, "text": paragraph})
        para_buf = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 代码块 fence 开始/结束
        fence_match = re.match(r"^\s*(```+|~~~+)", line)
        if fence_match:
            if not in_code:
                flush_para()
                in_code = True
                code_fence = fence_match.group(1)[0]  # ` or ~
                para_buf.append(line)
                i += 1
                continue
            if stripped.startswith(code_fence * 3):
                para_buf.append(line)
                flush_para()
                in_code = False
                code_fence = ""
                i += 1
                continue

        if in_code:
            para_buf.append(line)
            i += 1
            continue

        # 空行切分段落
        if not stripped:
            flush_para()
            i += 1
            continue

        # ATX 标题
        m = ATX_HEADING_RE.match(line)
        if m:
            flush_para()
            level = len(m.group(1))
            title = _normalize_heading_text(m.group(2))
            if title:
                blocks.append({"type": "heading", "level": level, "text": title})
            i += 1
            continue

        # setext 标题（当前行 + 下一行 === / ---）
        if not para_buf and i + 1 < n:
            next_line = lines[i + 1]
            m2 = SETEXT_RE.match(next_line)
            if m2 and stripped:
                level = 1 if m2.group(1).startswith("=") else 2
                blocks.append({"type": "heading", "level": level, "text": stripped})
                i += 2
                continue

        para_buf.append(line)
        i += 1

    flush_para()
    return blocks


def update_heading_stack(
    stack: list[tuple[int, str]], new_level: int, new_text: str
) -> list[tuple[int, str]]:
    """弹出层级 >= new_level 的旧标题，压入新标题"""
    stack = [(lvl, txt) for lvl, txt in stack if lvl < new_level]
    stack.append((new_level, new_text))
    return stack


def headings_str(stack: list[tuple[int, str]]) -> str:
    return " > ".join(txt for _, txt in stack)


def chunk_md(text: str) -> list[dict]:
    """
    将 MD 全文转为 chunk 列表：
      [{"headings": str, "content": str}, ...]
    """
    text = truncate_tail(text)
    blocks = parse_md_blocks(text)

    heading_stack: list[tuple[int, str]] = []
    buf = ""
    buf_headings = ""
    results = []

    def flush(headings: str, content: str) -> None:
        content = content.strip()
        if content:
            results.append({"headings": headings, "content": content})

    for block in blocks:
        if block["type"] == "heading":
            flush(buf_headings, buf)
            buf = ""
            heading_stack = update_heading_stack(
                heading_stack, block["level"], block["text"]
            )
            buf_headings = headings_str(heading_stack)
            continue

        para = block["text"]
        if len(buf) + len(para) + 1 > MAX_CHARS:
            prev = buf
            flush(buf_headings, prev)
            buf = _overlap_tail(prev, OVERLAP_CHARS)

            candidate = (buf + "\n" + para).strip() if buf else para
            pieces = split_long_text(candidate, MAX_CHARS, OVERLAP_CHARS)
            for piece in pieces[:-1]:
                flush(buf_headings, piece)
            buf = pieces[-1] if pieces else ""
        else:
            buf = (buf + "\n" + para).strip() if buf else para

    flush(buf_headings, buf)
    return results


# ─── 文件处理 ─────────────────────────────────────────────────────────────────

def process_md(md_path: Path) -> list[dict]:
    """处理单个 MD 文件，返回语义分段列表"""
    log.info(f"  读取: {md_path.name}")
    try:
        text = md_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = md_path.read_text(encoding="gbk", errors="replace")

    chunks_raw = chunk_md(text)
    log.info(f"  原始 chunk 数: {len(chunks_raw)}")

    # 约定：
    #   title    = 文件名（不含后缀）
    #   source   = 相对 origin_file 的路径（含后缀）
    #   category = 一级分类（如 Diabetes）
    title = md_path.stem
    try:
        rel = md_path.resolve().relative_to(ORIGIN_DIR.resolve())
        parts = rel.parts
        # MinerU 常见结构：<category>/<doc_name>/full.md
        # 这种情况下 source 统一回写为 <category>/<doc_name>.pdf
        if md_path.name.lower() == "full.md" and len(parts) >= 2:
            title = md_path.parent.name
            source = f"{Path(*parts[:-1]).as_posix()}.pdf"
        else:
            source = rel.as_posix()
        if len(parts) >= 2:
            category = parts[0]
        else:
            category = md_path.parent.name
    except ValueError:
        category = md_path.parent.name
        source = md_path.name
    chunks_out = []

    for i, c in enumerate(chunks_raw):
        if not c["content"]:
            continue
        chunks_out.append(
            {
                "chunk_id": i,
                "title": title,
                "source": source,
                "category": category,
                "headings": c["headings"],
                "content": c["content"],
            }
        )

    log.info(f"  有效 chunk 数: {len(chunks_out)}")
    log_chunk_stats([c["content"] for c in chunks_out])
    return chunks_out


def resolve_single_output_path(md_path: Path) -> Path:
    """
    单文件保存路径：
      - 若文件位于 ORIGIN_DIR 下：保持相对目录结构
      - 否则保存到 processed_md/_single/
    """
    try:
        rel = md_path.resolve().relative_to(ORIGIN_DIR.resolve())
        out_dir = OUTPUT_DIR / rel.parent
    except ValueError:
        out_dir = OUTPUT_DIR / "_single"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{md_path.stem}.json"


# ─── 批量处理 ─────────────────────────────────────────────────────────────────

def batch_process(force: bool = False):
    """
    force=True 时强制重新处理已有结果的 MD
    """
    if not ORIGIN_DIR.exists():
        log.error(f"原始文件目录不存在: {ORIGIN_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(ORIGIN_DIR.rglob("*.md"))
    log.info(f"发现 {len(md_files)} 个 MD 文件\n")

    all_index = []

    for md_path in md_files:
        rel = md_path.relative_to(ORIGIN_DIR)
        out_dir = OUTPUT_DIR / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{md_path.stem}.json"

        if out_path.exists() and not force:
            log.info(f"[跳过] 已处理: {rel}")
            with open(out_path, encoding="utf-8") as f:
                all_index.extend(json.load(f))
            continue

        log.info(f"{'─' * 60}")
        log.info(f"处理: {rel}")
        t0 = time.time()

        chunks = process_md(md_path)

        if chunks:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            log.info(f"  保存 -> {out_path}  ({time.time() - t0:.1f}s)")
            all_index.extend(chunks)
        else:
            log.warning(f"  无输出，跳过保存  ({time.time() - t0:.1f}s)")

    index_path = OUTPUT_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_index, f, ensure_ascii=False, indent=2)

    log.info(f"\n{'=' * 60}")
    log.info(f"全部完成: {len(md_files)} 个 MD -> {len(all_index)} 个 chunk")
    log.info(f"汇总索引 -> {index_path}")


# ─── 单文件诊断入口 ───────────────────────────────────────────────────────────

def diagnose(md_path: str):
    """对单个 MD 进行诊断，打印前 5 个 chunk 内容"""
    path = Path(md_path)
    chunks = process_md(path)
    print(f"\n共 {len(chunks)} 个 chunk\n")
    for c in chunks[:5]:
        print(f"[headings={c['headings']!r}]")
        print(c["content"][:300])
        print("---")


def process_single(md_path: str, save: bool):
    """单文件处理，可选保存"""
    path = Path(md_path)
    if not path.exists():
        log.error(f"文件不存在: {path}")
        return

    chunks = process_md(path)
    if not save:
        print(f"\n共 {len(chunks)} 个 chunk（诊断模式，不保存）\n")
        return

    out_path = resolve_single_output_path(path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    log.info(f"单文件保存 -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="批量分段 Markdown 文档")
    parser.add_argument("md_path", nargs="?", help="单个 md 文件路径（不填则批量处理）")
    parser.add_argument("--save", action="store_true", help="单文件模式下保存 JSON")
    parser.add_argument("--force", action="store_true", help="批量模式强制重跑已有结果")
    return parser.parse_args()


# ─── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.md_path:
        process_single(args.md_path, save=args.save)
    else:
        batch_process(force=args.force)
