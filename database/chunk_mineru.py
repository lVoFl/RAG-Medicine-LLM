"""
将 MinerU 解压结果（processed_pdf/_extracted）做分段，输出 RAG 可用 JSON。

输入：
  - 优先读取每篇文档目录下的 content_list_v2.json
  - 若不存在则回退读取 full.md

输出：
  - processed_mineru_chunks/<category>/<doc_name>.json
  - processed_mineru_chunks/index.json

用法：
  python chunk_mineru.py
  python chunk_mineru.py --max-chars 900 --overlap-chars 120
  python chunk_mineru.py --force
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

INPUT_DIR = Path(__file__).parent / "processed_pdf" / "_extracted"
ORIGIN_DIR = Path(__file__).parent / "origin_file"
OUTPUT_DIR = Path(__file__).parent / "processed_mineru_chunks"

MAX_CHARS = 900
MIN_CHARS = 120
OVERLAP_CHARS = 120

SKIP_BLOCK_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
    "page_footnote",
    "image",
}

SKIP_NODE_KEYS = {
    "type",
    "bbox",
    "poly",
    "position",
    "image_source",
    "path",
    "table_type",
    "table_nest_level",
    "list_type",
}


def remove_pdf_spaces(text: str) -> str:
    text = re.sub(
        r"(?<![a-zA-Z\d])([a-zA-Z\d])(?: ([a-zA-Z\d]))+(?![a-zA-Z\d])",
        lambda m: m.group(0).replace(" ", ""),
        text,
    )
    text = re.sub(r"([A-Z]) ([a-z])(?![a-zA-Z])", r"\1\2", text)
    text = re.sub(r"(\d+) \. (\d+)", r"\1.\2", text)
    text = re.sub(r"\( ", "(", text)
    text = re.sub(r" \)", ")", text)
    text = re.sub(r"' ", "'", text)
    text = re.sub(r" '", "'", text)
    text = re.sub(r" ([，。、；：！？‰°])", r"\1", text)
    text = re.sub(r" %", "%", text)
    text = re.sub(r" ,", ",", text)
    text = re.sub(
        r"\[ ?(\d+(?:[, ]+\d+)*) ?\]",
        lambda m: "[" + re.sub(r"\s", "", m.group(1)) + "]",
        text,
    )
    text = re.sub(r" ([≥≤><≈]) ", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 清理 MinerU 结构噪声词（并非正文内容）
    text = re.sub(r"(?<![A-Za-z])(?:text|equation_inline)(?![A-Za-z])\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def strip_html(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|tr|li|h\d)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    return remove_pdf_spaces(text)


def collect_text_from_node(node: Any) -> list[str]:
    out: list[str] = []
    if node is None:
        return out
    if isinstance(node, str):
        s = node.strip()
        if s:
            out.append(s)
        return out
    if isinstance(node, (int, float)):
        out.append(str(node))
        return out
    if isinstance(node, list):
        for item in node:
            out.extend(collect_text_from_node(item))
        return out
    if isinstance(node, dict):
        # 常见叶子结构：{"type": "...", "content": "..."}
        # 只取 content，避免把 type 标签词拼进正文。
        if "content" in node and set(node.keys()).issubset({"type", "content"}):
            return collect_text_from_node(node.get("content"))

        for key, value in node.items():
            if key in SKIP_NODE_KEYS:
                continue
            out.extend(collect_text_from_node(value))
        return out
    return out


def normalize_heading_level(raw_level: Any) -> int:
    try:
        lv = int(raw_level)
        return max(1, min(6, lv))
    except Exception:
        return 1


def update_heading_stack(stack: list[str], title: str, level: int) -> list[str]:
    level = max(1, level)
    while len(stack) < level:
        stack.append("")
    stack[level - 1] = title
    return [x for x in stack[:level] if x]


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = remove_pdf_spaces(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def page_range_str(pages: set[int]) -> str:
    if not pages:
        return ""
    lo = min(pages)
    hi = max(pages)
    return f"p{lo}" if lo == hi else f"p{lo}-p{hi}"


def build_category_map(origin_dir: Path) -> dict[str, str]:
    """
    filename -> category
    """
    mapping: dict[str, str] = {}
    if not origin_dir.exists():
        return mapping
    for pdf in origin_dir.rglob("*.pdf"):
        mapping[pdf.name] = pdf.parent.name
    return mapping


def parse_content_list_v2(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []

    sections: list[dict[str, Any]] = []
    heading_stack: list[str] = []
    current: dict[str, Any] | None = None

    def flush_current() -> None:
        nonlocal current
        if current and current["texts"]:
            sections.append(current)
        current = None

    for page_idx, page_blocks in enumerate(data, start=1):
        if not isinstance(page_blocks, list):
            continue

        for block in page_blocks:
            if not isinstance(block, dict):
                continue

            block_type = str(block.get("type", "")).strip().lower()
            content = block.get("content", {})

            if block_type in SKIP_BLOCK_TYPES:
                continue

            if block_type == "title":
                title_text = remove_pdf_spaces(" ".join(collect_text_from_node(content)))
                if not title_text:
                    continue
                level = normalize_heading_level(content.get("level", 1) if isinstance(content, dict) else 1)
                heading_stack = update_heading_stack(heading_stack, title_text, level)
                flush_current()
                continue

            if block_type == "table":
                table_parts: list[str] = []
                if isinstance(content, dict):
                    table_parts.extend(collect_text_from_node(content.get("table_caption")))
                    table_parts.extend(collect_text_from_node(content.get("table_footnote")))
                    html = content.get("html")
                    if isinstance(html, str) and html.strip():
                        table_parts.append(strip_html(html))
                text = remove_pdf_spaces("\n".join(table_parts))
            else:
                text = remove_pdf_spaces(" ".join(collect_text_from_node(content)))

            if not text:
                continue

            heading_key = " > ".join(heading_stack)
            if current is None or current["heading"] != heading_key:
                flush_current()
                current = {
                    "heading": heading_key,
                    "texts": [],
                    "pages": set(),
                }
            current["texts"].append(text)
            current["pages"].add(page_idx)

    flush_current()
    return sections


def parse_full_md(path: Path) -> list[dict[str, Any]]:
    text = remove_pdf_spaces(path.read_text(encoding="utf-8"))
    if not text:
        return []
    return [{"heading": "", "texts": [text], "pages": set()}]


def split_sections_to_chunks(
    sections: list[dict[str, Any]],
    max_chars: int,
    min_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = 0
    for sec in sections:
        content = remove_pdf_spaces("\n".join(sec.get("texts", [])))
        if not content:
            continue
        parts = chunk_text(content, max_chars=max_chars, overlap_chars=overlap_chars)
        for piece in parts:
            if len(piece) < min_chars and out:
                out[-1]["content"] = remove_pdf_spaces(out[-1]["content"] + "\n" + piece)
                continue
            out.append(
                {
                    "chunk_id": idx,
                    "headings": sec.get("heading", ""),
                    "content": piece,
                    "page_range": page_range_str(set(sec.get("pages", set()))),
                }
            )
            idx += 1
    return out


def process_doc_dir(
    doc_dir: Path,
    category_map: dict[str, str],
    max_chars: int,
    min_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    cpath = doc_dir / "content_list_v2.json"
    mpath = doc_dir / "full.md"

    sections: list[dict[str, Any]] = []
    if cpath.exists():
        sections = parse_content_list_v2(cpath)
    elif mpath.exists():
        sections = parse_full_md(mpath)

    chunks = split_sections_to_chunks(
        sections,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_chars=overlap_chars,
    )

    source_name = f"{doc_dir.name}.pdf"
    category = category_map.get(source_name, "unknown")
    for c in chunks:
        c["source"] = source_name
        c["category"] = category
    return chunks


def batch_chunk(
    input_dir: Path,
    output_dir: Path,
    origin_dir: Path,
    max_chars: int,
    min_chars: int,
    overlap_chars: int,
    force: bool,
) -> None:
    if not input_dir.exists():
        raise RuntimeError(f"输入目录不存在: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    category_map = build_category_map(origin_dir)

    doc_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    print(f"发现 {len(doc_dirs)} 篇文档目录")

    all_chunks: list[dict[str, Any]] = []
    done = 0

    for doc_dir in doc_dirs:
        source_name = f"{doc_dir.name}.pdf"
        category = category_map.get(source_name, "unknown")
        out_dir = output_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc_dir.name}.json"

        if out_path.exists() and not force:
            print(f"[跳过] 已存在: {out_path}")
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                all_chunks.extend(cached)
            continue

        chunks = process_doc_dir(
            doc_dir=doc_dir,
            category_map=category_map,
            max_chars=max_chars,
            min_chars=min_chars,
            overlap_chars=overlap_chars,
        )
        out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        all_chunks.extend(chunks)
        done += 1
        print(f"[完成] {doc_dir.name}: {len(chunks)} chunks -> {out_path}")

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"总计完成 {done} 篇，输出 chunk {len(all_chunks)} 条")
    print(f"汇总索引: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--origin-dir", type=Path, default=ORIGIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=OVERLAP_CHARS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    batch_chunk(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        origin_dir=args.origin_dir,
        max_chars=max(100, args.max_chars),
        min_chars=max(20, args.min_chars),
        overlap_chars=max(0, args.overlap_chars),
        force=args.force,
    )


if __name__ == "__main__":
    main()
