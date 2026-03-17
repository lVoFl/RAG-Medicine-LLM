"""
批量处理 origin_file 中的 TXT 文档，进行语义分段，
输出 JSON 供 RAG 检索使用。

docling 不支持 TXT 输入，本脚本使用自定义解析：
  1. 按空行切分段落
  2. 识别标题行（短行 + 中文编号模式），维护层级标题栈
  3. 按 MAX_CHARS 合并或拆分段落形成 chunk

输出目录结构：
  database/processed_txt/
    ├── Hypertension/
    │   ├── xxx.json          # 每个 TXT 的分段结果
    │   └── ...
    └── index.json            # 所有文档的汇总索引
"""

import json
import re
import time
import logging
from pathlib import Path

# ─── 配置 ────────────────────────────────────────────────────────────────────

ORIGIN_DIR = Path(__file__).parent / "origin_file"
OUTPUT_DIR = Path(__file__).parent / "processed_txt"

# 字符数上限（中文约 1.5 字/token，512 tokens ≈ 768 字）
MAX_CHARS = 768
# 视为标题的最大行长
HEADING_MAX_LEN = 40
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
    text = re.sub(r"DOI:.*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─── 标题识别 ─────────────────────────────────────────────────────────────────

# 中文显式层级标题模式（匹配优先级从高到低）
_HEADING_PATTERNS = [
    # 一级：汉字数字 + 顿号，如 "一、定义"
    (1, re.compile(r'^[一二三四五六七八九十百]+[、．.]')),
    # 二级：括号汉字数字，如 "（一）病因"
    (2, re.compile(r'^[（(][一二三四五六七八九十]+[）)]')),
    # 三级：阿拉伯数字+点，如 "1." "1．"
    (3, re.compile(r'^\d+[.．]\s')),
    # 四级：括号阿拉伯数字，如 "(1)" "（1）"
    (4, re.compile(r'^[（(]\d+[）)]')),
]


def detect_heading_level(line: str) -> int:
    """
    返回标题层级 1~4，0 表示非标题。
    无显式编号的短行（≤40 字，不以句子级标点结尾）视为二级标题，
    可嵌套在最近一个一级标题下（如 "遗传因素" 置于 "三、病因" 之下）。
    """
    line = line.strip()
    if not line or len(line) > HEADING_MAX_LEN:
        return 0
    for level, pat in _HEADING_PATTERNS:
        if pat.match(line):
            return level
    # 短行且不以句子终止标点结尾 → 视为隐式二级标题
    if len(line) <= HEADING_MAX_LEN and not re.search(r'[，。；！？,;!?]$', line):
        return 2
    return 0


# ─── 分段逻辑 ─────────────────────────────────────────────────────────────────

def parse_blocks(text: str) -> list[dict]:
    """
    将纯文本解析为 block 列表：
      {"type": "heading"|"paragraph", "level": int, "text": str}

    按 1 个以上连续空行切分段落；单行段落若检测为标题则单独成块，
    多行段落首行若为标题则拆开。
    """
    blocks = []
    raw_paras = re.split(r'\n\s*\n', text.strip())
    for raw in raw_paras:
        raw = raw.strip()
        if not raw:
            continue
        lines = raw.splitlines()
        if len(lines) == 1:
            level = detect_heading_level(lines[0])
            if level > 0:
                blocks.append({"type": "heading", "level": level, "text": lines[0].strip()})
            else:
                cleaned = clean_paragraph(lines[0])
                if cleaned:
                    blocks.append({"type": "paragraph", "level": 0, "text": cleaned})
        else:
            # 多行：首行可能是标题
            level = detect_heading_level(lines[0])
            if level > 0:
                blocks.append({"type": "heading", "level": level, "text": lines[0].strip()})
                rest = clean_paragraph("\n".join(lines[1:]))
                if rest:
                    blocks.append({"type": "paragraph", "level": 0, "text": rest})
            else:
                cleaned = clean_paragraph(raw)
                if cleaned:
                    blocks.append({"type": "paragraph", "level": 0, "text": cleaned})
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


def split_long_text(text: str, max_chars: int) -> list[str]:
    """按句子边界拆分超长文本"""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[。！？.!?])', text)
    chunks, buf = [], ""
    for sent in sentences:
        if len(buf) + len(sent) > max_chars and buf:
            chunks.append(buf.strip())
            buf = sent
        else:
            buf += sent
    if buf.strip():
        chunks.append(buf.strip())
    return chunks or [text]


def chunk_txt(text: str) -> list[dict]:
    """
    将 TXT 全文转为 chunk 列表：
      [{"headings": str, "content": str}, ...]

    合并策略：同一顶层标题下的各小节内容优先合并为一个 chunk，
    仅当 buffer 超出 MAX_CHARS 或遇到同级/更高层级的标题时才切断。
    这样 "三、病因 > 遗传因素"、"三、病因 > 精神紧张" 等短节
    会被合并到同一个 "三、病因" chunk 中。
    """
    text = truncate_tail(text)
    blocks = parse_blocks(text)

    heading_stack: list[tuple[int, str]] = []
    buf = ""
    buf_headings = ""
    results = []

    def flush(headings: str, content: str) -> None:
        content = content.strip()
        if content:
            results.append({"headings": headings, "content": content})

    def top_level() -> int:
        """返回当前栈中最高层级（数字最小）的层级号，栈空时返回 0"""
        return min(lvl for lvl, _ in heading_stack) if heading_stack else 0

    for block in blocks:
        if block["type"] == "heading":
            new_level = block["level"]
            if not heading_stack or new_level <= top_level():
                # 同级或更高层级的新节 → flush 当前 buffer，重置标题
                flush(buf_headings, buf)
                buf = ""
                heading_stack = update_heading_stack(
                    heading_stack, new_level, block["text"]
                )
                buf_headings = headings_str(heading_stack)
            else:
                # 更深的子标题 → 不 flush，仅更新栈；
                # buf_headings 保持父级，内容继续合并
                heading_stack = update_heading_stack(
                    heading_stack, new_level, block["text"]
                )
        else:
            para = block["text"]
            if len(buf) + len(para) + 1 > MAX_CHARS:
                flush(buf_headings, buf)
                buf = ""
                # 单段落本身超长则继续拆分
                pieces = split_long_text(para, MAX_CHARS)
                for piece in pieces[:-1]:
                    flush(buf_headings, piece)
                buf = pieces[-1] if pieces else ""
            else:
                buf = (buf + "\n" + para).strip() if buf else para

    flush(buf_headings, buf)
    return results


# ─── 文件处理 ─────────────────────────────────────────────────────────────────

def process_txt(txt_path: Path) -> list[dict]:
    """处理单个 TXT 文件，返回语义分段列表"""
    log.info(f"  读取: {txt_path.name}")
    try:
        text = txt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = txt_path.read_text(encoding="gbk", errors="replace")

    chunks_raw = chunk_txt(text)
    log.info(f"  原始 chunk 数: {len(chunks_raw)}")

    category = txt_path.parent.name
    source = txt_path.name
    chunks_out = []

    for i, c in enumerate(chunks_raw):
        if not c["content"]:
            continue
        chunks_out.append({
            "chunk_id": i,
            "source":   source,
            "category": category,
            "headings": c["headings"],
            "content":  c["content"],
        })

    log.info(f"  有效 chunk 数: {len(chunks_out)}")
    return chunks_out


# ─── 批量处理 ─────────────────────────────────────────────────────────────────

def batch_process(force: bool = False):
    """
    force=True 时强制重新处理已有结果的 TXT
    """
    if not ORIGIN_DIR.exists():
        log.error(f"原始文件目录不存在: {ORIGIN_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(ORIGIN_DIR.rglob("*.txt"))
    log.info(f"发现 {len(txt_files)} 个 TXT 文件\n")

    all_index = []

    for txt_path in txt_files:
        rel = txt_path.relative_to(ORIGIN_DIR)
        out_dir = OUTPUT_DIR / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (txt_path.stem + ".json")

        if out_path.exists() and not force:
            log.info(f"[跳过] 已处理: {rel}")
            with open(out_path, encoding="utf-8") as f:
                all_index.extend(json.load(f))
            continue

        log.info(f"{'─'*60}")
        log.info(f"处理: {rel}")
        t0 = time.time()

        chunks = process_txt(txt_path)

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
    log.info(f"全部完成: {len(txt_files)} 个 TXT -> {len(all_index)} 个 chunk")
    log.info(f"汇总索引 -> {index_path}")


# ─── 单文件诊断入口 ───────────────────────────────────────────────────────────

def diagnose(txt_path: str):
    """对单个 TXT 进行诊断，打印前 5 个 chunk 内容"""
    path = Path(txt_path)
    chunks = process_txt(path)
    print(f"\n共 {len(chunks)} 个 chunk\n")
    for c in chunks[:5]:
        print(f"[headings={c['headings']!r}]")
        print(c["content"][:300])
        print("---")


# ─── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # python batch_txt.py <txt_path>  -> 诊断模式
        diagnose(sys.argv[1])
    else:
        # 正常批量处理（force=True 重新处理所有）
        batch_process(force=True)
