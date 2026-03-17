import fitz  # PyMuPDF
import re
from collections import defaultdict
import json

def clean_medical_text(text):
    # 1️⃣ 去除多余空行
    text = re.sub(r'\n{2,}', '\n\n', text)

    # 2️⃣ 修复英文断行（上一行以字母结尾，下一行以小写字母开头）
    text = re.sub(r'([a-zA-Z,])\n([a-z])', r'\1 \2', text)

    # 3️⃣ 修复 Publishing \n House 这种情况
    text = re.sub(r'([A-Za-z])\n([A-Za-z])', r'\1 \2', text)

    # 4️⃣ 修复 Email 换行
    text = re.sub(r'Email：\n', 'Email: ', text)
    text = re.sub(r'Email:\n', 'Email: ', text)

    # 5️⃣ 去掉行尾多余空格
    text = re.sub(r'[ \t]+\n', '\n', text)
    
    return text

def extract_strict_two_column(pdf_path, margin=20, header_threshold=0.8):
    """
    两栏 PDF 提取 + 自动去除页眉页脚
    margin: 中线安全边界
    header_threshold: 统计重复页眉出现比例
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    # 第一步：统计每页最上方文本块（用于识别页眉）
    top_blocks_counter = defaultdict(int)
    for page in doc:
        blocks = page.get_text("blocks")
        if not blocks:
            continue
        blocks.sort(key=lambda b: b[1])  # 按 y0 排序
        top_block_text = blocks[0][4].strip()
        if top_block_text:
            top_blocks_counter[top_block_text] += 1

    # 重复出现超过阈值的块认为是页眉
    num_pages = len(doc)
    header_candidates = set(
        text for text, count in top_blocks_counter.items() if count / num_pages >= header_threshold
    )

    # 第二步：逐页提取
    for page_index in range(len(doc)):
        page = doc[page_index]
        blocks = page.get_text("blocks")
        page_width = page.rect.width
        mid_x = page_width / 2

        left_blocks = []
        right_blocks = []

        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            text = text.strip()
            if not text:
                continue

            # 跳过页眉
            if text in header_candidates:
                continue

            # 跳过页面顶部的页码块（右上角 · 22· 格式）
            page_height = page.rect.height
            if y0 < page_height * 0.12 and re.search(r'[··•・]\s*\d+\s*[··•・]?', text):
                continue

            # 中线中心
            center_x = (x0 + x1) / 2

            if center_x < mid_x - margin:
                left_blocks.append((y0, text))
            elif center_x > mid_x + margin:
                right_blocks.append((y0, text))
            else:
                # 中线附近（页码、奇怪的块）丢弃
                continue

        # 各自按 y0 排序
        left_blocks.sort(key=lambda x: x[0])
        right_blocks.sort(key=lambda x: x[0])

        page_text = f"\n\n===== 第 {page_index+1} 页 =====\n"

        # 拼接左栏
        for _, text in left_blocks:
            page_text += text + "\n"

        # 拼接右栏
        for _, text in right_blocks:
            page_text += text + "\n"

        full_text += page_text

    return full_text

def smart_paragraph_merge(text):
    lines = text.split("\n")
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                merged.append(buffer)
                buffer = ""
            continue

        # 一级/二级标题（一、 （一）等）：立即输出，不放入 buffer，
        # 防止后续正文行因 buffer 不以句末标点结尾而被错误合并进标题
        if re.match(r"^[一二三四五六七八九十]+、", line) \
           or re.match(r"^（[一二三四五六七八九十]+）", line):
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(line)
            continue

        # 三级数字标题（1. 2. 等）：放入 buffer，以便与跨行内容拼接
        if re.match(r"^\d+\.[^\d]", line):
            if buffer:
                merged.append(buffer)
            buffer = line
            continue

        if not buffer:
            buffer = line
        else:
            # 如果上一行不是完整句子，拼接
            if not re.search(r"[。！？；：]$", buffer):
                buffer += line
            else:
                merged.append(buffer)
                buffer = line

    if buffer:
        merged.append(buffer)

    return "\n".join(merged)

def remove_footer(text):
    text = re.sub(r"DOI:.*", "", text)
    text = re.sub(r"收稿日期.*", "", text)
    text = re.sub(r"引用本文.*", "", text)
    text = re.sub(r"·\s*\d+\s*·", "", text)
    text = re.sub(r"基层常见疾病合理用药指南", "", text)
    return text

def strong_medical_clean(text):

    # 0️⃣ 截断文末无关内容（参考文献、专家组名单、征订通知等）
    tail_markers = [
        '参考文献',
        '《基层医疗卫生机构合理用药指南》编写专家组',
        '心血管系统疾病合理用药指南编写组',
        '利益冲突',
    ]
    for marker in tail_markers:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]

    # 1️⃣ 删除 Base64 类异常串
    text = re.sub(r'[A-Za-z0-9+/]{15,}={0,2}\??', '', text)

    # 2️⃣ 删除页眉（中英文期刊头）
    text = re.sub(r'中华全科医师杂志.*?No\. 1', '', text)

    # 3️⃣ 删除页标
    text = re.sub(r'===== 第 \d+ 页 =====', '', text)

    # 4️⃣ 删除 DOI
    text = re.sub(r'DOI:.*', '', text)

    # 5️⃣ 删除页码符号（兼容不同 Unicode 中圆点：· · • ・）
    text = re.sub(r'[··•・·]{1,2}\s*\d+\s*[··•・·]{0,2}', '', text)

    # 6️⃣ 删除引用信息
    text = re.sub(r'引用本文.*', '', text)

    # 7️⃣ 删除收稿日期
    text = re.sub(r'收稿日期.*', '', text)

    # 8️⃣ 删除多余空行
    text = re.sub(r'\n{2,}', '\n\n', text)

    return text.strip()

def semantic_chunk(text, max_chunk_size=500, overlap=50):
    """
    将清洗后的医学文本按语义分段，用于向量数据库存储。

    策略：
    - 按标题层级（一、> （一）> 1.）划定 chunk 边界
    - 超长段落按中文句子边界（。！？）二次切割，带 overlap
    - 每个 chunk 记录完整标题路径（title）便于检索时携带上下文

    Args:
        text:           smart_paragraph_merge 处理后的文本
        max_chunk_size: 单个 chunk 最大字符数（默认 500）
        overlap:        相邻 chunk 的尾部重叠字符数（默认 50）

    Returns:
        List[dict]: [{'chunk_id': int, 'title': str, 'content': str}, ...]
    """
    HEADING_PATTERNS = [
        (1, re.compile(r'^[一二三四五六七八九十百]+、')),
        (2, re.compile(r'^（[一二三四五六七八九十百]+）')),
        (3, re.compile(r'^\d+\.[^\d]')),
    ]

    def get_level(line):
        for level, pat in HEADING_PATTERNS:
            if pat.match(line):
                return level
        return None

    def split_by_sentences(content, max_size, overlap_size):
        """按句子边界切割长段落，带尾部 overlap"""
        sentences = re.split(r'(?<=[。！？])', content)
        sentences = [s for s in sentences if s.strip()]
        result = []
        buf = ''
        for sent in sentences:
            if len(buf) + len(sent) <= max_size:
                buf += sent
            else:
                if buf:
                    result.append(buf.strip())
                tail = buf[-overlap_size:] if overlap_size and buf else ''
                buf = tail + sent
        if buf.strip():
            result.append(buf.strip())
        return result if result else [content.strip()]

    lines = text.split('\n')
    chunks = []
    chunk_id = 0
    title_stack = []   # [(level, heading_text), ...]
    content_buf = []

    def flush():
        nonlocal chunk_id
        content = '\n'.join(content_buf).strip()
        content_buf.clear()
        if not content:
            return
        full_title = ' > '.join(t for _, t in title_stack)
        if len(content) <= max_chunk_size:
            chunks.append({
                'chunk_id': chunk_id,
                'title': full_title,
                'content': content,
            })
            chunk_id += 1
        else:
            for sub in split_by_sentences(content, max_chunk_size, overlap):
                chunks.append({
                    'chunk_id': chunk_id,
                    'title': full_title,
                    'content': sub,
                })
                chunk_id += 1

    for line in lines:
        line = line.strip()
        if not line:
            continue
        level = get_level(line)
        if level is not None:
            flush()
            # 弹出同级及更深层级的标题，保留上层标题
            title_stack = [(l, t) for l, t in title_stack if l < level]
            # 三级标题（如 "8.药物相互作用：..."）可能因行合并而包含正文内容，
            # 只截取标题标识部分（到第一个"："为止），其余内容放回正文
            if level == 3:
                colon_idx = line.find('：')
                if colon_idx > 0 and len(line) > colon_idx + 5:
                    heading = line[:colon_idx + 1]
                    remainder = line[colon_idx + 1:].strip()
                    title_stack.append((level, heading))
                    if remainder:
                        content_buf.append(remainder)
                else:
                    title_stack.append((level, line))
            else:
                title_stack.append((level, line))
        else:
            content_buf.append(line)

    flush()
    return chunks


if __name__ == "__main__":
    # pdf_path = "C:\\Users\\asus\\Desktop\\【用药助手】高血压基层合理用药指南.pdf"
    pdf_path = './database/origin_file/Hypertension/【用药助手】中国老年高血压管理指南（2023）.pdf'
    text = extract_strict_two_column(pdf_path)
    text = strong_medical_clean(text)
    text = remove_footer(text)
    text = smart_paragraph_merge(text)

    chunks = semantic_chunk(text, max_chunk_size=500, overlap=50)
    for chunk in chunks[:5]:
        print(f"[{chunk['chunk_id']}] {chunk['title']}")
        print(chunk['content'])
        print("---")
    print(chunks)
    with open('./1.json', "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)