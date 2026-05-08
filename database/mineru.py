"""
MinerU 批量 PDF 解析脚本（按真实返回结构 data.extract_result 实现）

流程：
1. 扫描 origin_file 下全部 PDF
2. 申请批量上传地址并上传
3. 轮询 /extract-results/batch/{batch_id}
4. 保存结果 JSON
5. 可选下载 full_zip_url 到本地

示例：
  python mineru.py --token "your_token"
  python mineru.py --token "your_token" --download-zip
  python mineru.py --token "your_token" --batch-id 8714c77b-f191-452b-be7a-c90356732702 --download-zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://mineru.net/api/v4"
DEFAULT_MODEL_VERSION = "vlm"
DEFAULT_TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIxNjQwMDI1MSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3NjU5NDYyNiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiZGNlY2I4NmUtMTk3Ny00NTM0LWE2MmYtMGYzMmM3NzI4NDk0IiwiZW1haWwiOiIiLCJleHAiOjE3ODQzNzA2MjZ9.XDmEyRwH_28UITLMhLHrBNh0HXJjFQoTWQjg3I01VZ5VX91z_Qte7tr7OeL5cl5WNkGwimY52aA9PJGSeePiYQ"  # 可直接填 token；命令行 --token 会覆盖
VALID_EXTENSIONS = {".md", ".markdown", ".txt", ".json"}


def build_headers(token: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def collect_pdfs(origin_dir: Path) -> list[Path]:
    return sorted(origin_dir.rglob("*.pdf"))


def truncate_utf8(text: str, max_bytes: int) -> str:
    """按 UTF-8 字节截断，保证不切坏字符。"""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def build_data_id(pdf: Path, origin_dir: Path, max_len: int = 128) -> str:
    """
    生成符合 MinerU 限制的 data_id（<= max_len）。
    优先使用相对路径；超长时保留一级分类 + 文件名主体，并追加哈希保证唯一。
    """
    rel = pdf.relative_to(origin_dir).as_posix()
    if len(rel.encode("utf-8")) <= max_len:
        return rel

    rel_parts = pdf.relative_to(origin_dir).parts
    category = rel_parts[0] if len(rel_parts) > 1 else "root"
    stem = pdf.stem
    suffix = pdf.suffix or ".pdf"
    digest = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]

    # 目标格式：category/stem__digest.suffix（按字节限长）
    # 先给 category 限一个较小上限，避免分类名过长吞掉预算。
    category_safe = truncate_utf8(category, 32) or "root"

    tail = f"__{digest}{suffix}"
    prefix = f"{category_safe}/"
    remain_bytes = max_len - len(prefix.encode("utf-8")) - len(tail.encode("utf-8"))
    if remain_bytes < 1:
        # 极端兜底：仅保留 hash
        fallback = f"f/{digest}{suffix}"
        return truncate_utf8(fallback, max_len)

    stem_safe = truncate_utf8(stem, remain_bytes) or "f"
    candidate = f"{prefix}{stem_safe}{tail}"
    if len(candidate.encode("utf-8")) <= max_len:
        return candidate

    # 再兜底一次，保证绝对不超
    return truncate_utf8(candidate, max_len)


def filter_unprocessed_pdfs(files: list[Path], origin_dir: Path, prepared_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    基于 origin_dir 的目录结构判断是否已处理：
      origin_dir/<pdf相对父目录>/ 下已存在与 pdf 去后缀名相同（或前缀匹配）的目录
    若该目录已存在且非空，则视为已处理并跳过。
    """
    pending: list[Path] = []
    skipped: list[Path] = []

    for pdf in files:
        rel = pdf.relative_to(origin_dir)
        # 按你的需求：在 origin_file 目录内判断是否已有同名产物目录
        parent_dir = origin_dir / rel.parent
        stem = pdf.stem

        matched = False
        if parent_dir.exists():
            for child in parent_dir.iterdir():
                if not child.is_dir():
                    continue
                # 目录名与 PDF 去后缀名完全一致，或以前缀形式一致（如后面拼了标识）
                if child.name == stem or child.name.startswith(stem):
                    if any(child.iterdir()):
                        matched = True
                        break

        if matched:
            skipped.append(pdf)
        else:
            pending.append(pdf)

    return pending, skipped


def make_batch_payload(files: list[Path], origin_dir: Path, model_version: str) -> dict[str, Any]:
    items = []
    for pdf in files:
        data_id = build_data_id(pdf, origin_dir)
        items.append({"name": pdf.name, "data_id": data_id})
    return {"files": items, "model_version": model_version}


def apply_upload_urls(token: str, payload: dict[str, Any]) -> tuple[str, list[str]]:
    url = f"{BASE_URL}/file-urls/batch"
    resp = requests.post(url, headers=build_headers(token), json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    if result.get("code") != 0:
        raise RuntimeError(f"申请上传地址失败: {result.get('msg')}")

    data = result.get("data", {})
    batch_id = data.get("batch_id")
    file_urls = data.get("file_urls", [])
    if not batch_id:
        raise RuntimeError("返回缺少 batch_id")
    if not isinstance(file_urls, list) or not file_urls:
        raise RuntimeError("返回缺少 file_urls")
    return batch_id, file_urls


def upload_files(files: list[Path], upload_urls: list[str]) -> None:
    if len(files) != len(upload_urls):
        raise RuntimeError(f"文件数({len(files)})与上传 URL 数({len(upload_urls)})不一致")

    for i, (pdf, put_url) in enumerate(zip(files, upload_urls), start=1):
        with pdf.open("rb") as f:
            resp = requests.put(put_url, data=f, timeout=600)
        if resp.status_code != 200:
            raise RuntimeError(f"上传失败: {pdf} -> HTTP {resp.status_code}")
        print(f"[{i}/{len(files)}] 上传成功: {pdf}")


def parse_extract_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    只解析 MinerU 当前真实结构：
    {
      "code": 0,
      "data": {
        "batch_id": "...",
        "extract_result": [{...}]
      }
    }
    """
    data = result.get("data")
    if not isinstance(data, dict):
        return []
    items = data.get("extract_result")
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def normalize_state(state: Any) -> str:
    return str(state or "").strip().lower()


def is_terminal_state(state: str) -> bool:
    return state in {"done", "failed", "error", "cancelled"}


def poll_batch_result(token: str, batch_id: str, timeout: int, interval: int) -> dict[str, Any]:
    url = f"{BASE_URL}/extract-results/batch/{batch_id}"
    start = time.time()
    last_result: dict[str, Any] = {}

    while True:
        resp = requests.get(url, headers=build_headers(token), timeout=60)
        resp.raise_for_status()
        result = resp.json()
        last_result = result

        if result.get("code") != 0:
            raise RuntimeError(f"查询失败: {result.get('msg')}")

        items = parse_extract_result(result)
        elapsed = int(time.time() - start)

        if not items:
            print(f"[{elapsed}s] 暂未返回 extract_result，继续轮询...")
        else:
            states = [
                (
                    str(item.get("data_id") or item.get("file_name") or "unknown"),
                    normalize_state(item.get("state")),
                )
                for item in items
            ]
            done_count = sum(1 for _, st in states if is_terminal_state(st))
            print(f"[{elapsed}s] 进度: {done_count}/{len(states)}")
            for name, st in states[:10]:
                print(f"  - {name}: {st}")
            if len(states) > 10:
                print(f"  - ... 其余 {len(states) - 10} 个文件")

            if done_count == len(states):
                return result

        if time.time() - start >= timeout:
            print(f"轮询超时（{timeout}s），返回最后一次结果。")
            return last_result

        time.sleep(interval)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_filename_from_data_id(data_id: str, fallback: str) -> str:
    stem = Path(data_id).stem if data_id else Path(fallback).stem
    name = stem.strip() or "result"
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def safe_path_fragment(text: str) -> str:
    text = (text or "").strip().replace("\\", "/")
    parts = [p for p in text.split("/") if p and p not in {".", ".."}]
    cleaned: list[str] = []
    bad = '<>:"/\\|?*'
    for p in parts:
        item = p
        for ch in bad:
            item = item.replace(ch, "_")
        cleaned.append(item.strip() or "unknown")
    return "/".join(cleaned)


def download_full_zips(final_result: dict[str, Any], zip_dir: Path) -> tuple[int, int]:
    zip_dir.mkdir(parents=True, exist_ok=True)
    items = parse_extract_result(final_result)
    ok, skipped = 0, 0

    for item in items:
        state = normalize_state(item.get("state"))
        zip_url = str(item.get("full_zip_url") or "").strip()
        data_id = str(item.get("data_id") or "")
        file_name = str(item.get("file_name") or "")

        if state != "done" or not zip_url.startswith("http"):
            skipped += 1
            continue

        out_name = safe_filename_from_data_id(data_id, file_name) + ".zip"
        out_path = zip_dir / out_name

        resp = requests.get(zip_url, timeout=300)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        ok += 1
        print(f"已下载: {out_path}")

    return ok, skipped


def extract_zip_files(zip_dir: Path, extracted_dir: Path) -> int:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(zip_dir.glob("*.zip"))
    count = 0

    for zf in zip_files:
        subdir = extracted_dir / zf.stem
        subdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(subdir)
        count += 1
        print(f"已解压: {zf} -> {subdir}")

    return count


def copy_prepared_files(final_result: dict[str, Any], extracted_dir: Path, prepared_dir: Path) -> tuple[int, int]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    items = parse_extract_result(final_result)
    kept = 0
    skipped = 0

    for item in items:
        state = normalize_state(item.get("state"))
        if state != "done":
            skipped += 1
            continue

        data_id = str(item.get("data_id") or "")
        file_name = str(item.get("file_name") or "")
        zip_stem = safe_filename_from_data_id(data_id, file_name)
        src_root = extracted_dir / zip_stem
        if not src_root.exists():
            skipped += 1
            print(f"跳过（未找到解压目录）: {src_root}")
            continue

        relative_parent = Path(safe_path_fragment(Path(data_id).parent.as_posix() if data_id else "unknown"))
        dst_root = prepared_dir / relative_parent / zip_stem
        dst_root.mkdir(parents=True, exist_ok=True)

        current_kept = 0
        for p in src_root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in VALID_EXTENSIONS:
                continue
            target = dst_root / p.name
            if target.exists():
                target = dst_root / f"{p.stem}_{int(time.time() * 1000)}{p.suffix.lower()}"
            shutil.copy2(p, target)
            kept += 1
            current_kept += 1

        if current_kept == 0:
            skipped += 1
            print(f"跳过（无有效文件）: {zip_stem}")
        else:
            print(f"已整理 {current_kept} 个有效文件: {dst_root}")

    return kept, skipped


def query_batch_once(token: str, batch_id: str) -> dict[str, Any]:
    url = f"{BASE_URL}/extract-results/batch/{batch_id}"
    resp = requests.get(url, headers=build_headers(token), timeout=60)
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 0:
        raise RuntimeError(f"查询失败: {result.get('msg')}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=DEFAULT_TOKEN, help="MinerU API token")
    parser.add_argument("--origin-dir", type=Path, default=Path(__file__).parent / "origin_file")
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).parent / "finish_file")
    parser.add_argument("--zip-dir", type=Path, default=Path(__file__).parent / "processed_pdf")
    parser.add_argument("--extracted-dir", type=Path, default=Path(__file__).parent / "processed_pdf" / "_extracted")
    parser.add_argument("--prepared-dir", type=Path, default=Path(__file__).parent / "processed_txt")
    parser.add_argument("--model-version", type=str, default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--batch-id", type=str, default="", help="仅查询已有批次并可下载结果，不重新上传")
    parser.add_argument("--download-zip", action="store_true", help="下载 full_zip_url 到 zip-dir")
    parser.add_argument("--extract-zip", action="store_true", help="下载后自动解压 ZIP")
    parser.add_argument("--prepare-files", action="store_true", help="整理有效文本文件到 prepared-dir")
    args = parser.parse_args()

    token = (args.token or "").strip()
    if not token:
        raise RuntimeError("请通过 --token 提供 token，或填写 DEFAULT_TOKEN")

    if args.batch_id:
        batch_id = args.batch_id.strip()
        final_result = query_batch_once(token, batch_id)
        print(f"已查询 batch_id: {batch_id}")
    else:
        if not args.origin_dir.exists():
            raise RuntimeError(f"目录不存在: {args.origin_dir}")
        all_pdf_files = collect_pdfs(args.origin_dir)
        if not all_pdf_files:
            print(f"未找到 PDF: {args.origin_dir}")
            return

        pdf_files, skipped_files = filter_unprocessed_pdfs(all_pdf_files, args.origin_dir, args.prepared_dir)
        if skipped_files:
            print(f"已跳过 {len(skipped_files)} 个已处理 PDF（目标目录已存在）")
            for p in skipped_files[:10]:
                print(f"  - {p.relative_to(args.origin_dir)}")
            if len(skipped_files) > 10:
                print(f"  - ... 其余 {len(skipped_files) - 10} 个文件")

        if not pdf_files:
            print("待处理 PDF 数为 0（全部已处理），本次结束。")
            return

        print(f"发现 {len(all_pdf_files)} 个 PDF，本次待处理 {len(pdf_files)} 个，开始申请上传地址...")
        payload = make_batch_payload(pdf_files, args.origin_dir, args.model_version)
        batch_id, upload_urls = apply_upload_urls(token, payload)
        print(f"batch_id: {batch_id}")

        print("开始上传 PDF...")
        upload_files(pdf_files, upload_urls)
        print("上传完成，开始轮询结果...")
        final_result = poll_batch_result(token, batch_id, args.timeout, args.interval)

    data = final_result.get("data", {})
    batch_id = str(data.get("batch_id") or args.batch_id or "unknown")
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = args.result_dir / f"mineru_batch_{batch_id}_{ts}.json"
    save_json(json_path, final_result)
    print(f"结果已保存: {json_path}")

    if args.download_zip:
        ok, skipped = download_full_zips(final_result, args.zip_dir)
        print(f"ZIP 下载完成: 成功 {ok}，跳过 {skipped}")

    if args.extract_zip:
        n = extract_zip_files(args.zip_dir, args.extracted_dir)
        print(f"ZIP 解压完成：{n} 个")

    if args.prepare_files:
        if not args.extract_zip:
            print("提示：未启用 --extract-zip，将直接基于现有解压目录整理。")
        kept, skipped_items = copy_prepared_files(final_result, args.extracted_dir, args.prepared_dir)
        print(f"有效文件整理完成：保留 {kept} 个，跳过 {skipped_items} 个条目")


if __name__ == "__main__":
    main()
