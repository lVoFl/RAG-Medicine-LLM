"""
下载指定 MinerU batch_id 的处理结果（full_zip_url），并可自动解压、整理有效文本文件。

默认 batch_id:
  8714c77b-f191-452b-be7a-c90356732702

用法:
  python download_batch_8714.py --token "你的token"
  python download_batch_8714.py --token "你的token" --extract-zip
  python download_batch_8714.py --token "你的token" --extract-zip --prepare-files
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://mineru.net/api/v4"
DEFAULT_TOKEN = ""  # 可直接填；命令行 --token 会覆盖
DEFAULT_BATCH_ID = "8714c77b-f191-452b-be7a-c90356732702"
VALID_EXTENSIONS = {".md", ".markdown", ".txt", ".json"}


def build_headers(token: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def query_batch(token: str, batch_id: str) -> dict[str, Any]:
    url = f"{BASE_URL}/extract-results/batch/{batch_id}"
    resp = requests.get(url, headers=build_headers(token), timeout=60)
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 0:
        raise RuntimeError(f"查询失败: {result.get('msg')}")
    return result


def get_extract_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    data = result.get("data")
    if not isinstance(data, dict):
        return []
    items = data.get("extract_result")
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def safe_name_from_data_id(data_id: str, fallback: str) -> str:
    stem = Path(data_id).stem if data_id else Path(fallback).stem
    name = stem.strip() or "result"
    for ch in '<>:"/\\|?*':
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


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def download_full_zips(result: dict[str, Any], output_dir: Path) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = get_extract_items(result)
    success = 0
    skipped = 0

    for i, item in enumerate(items, start=1):
        state = str(item.get("state") or "").strip().lower()
        zip_url = str(item.get("full_zip_url") or "").strip()
        data_id = str(item.get("data_id") or "")
        file_name = str(item.get("file_name") or "")

        if state != "done" or not zip_url.startswith("http"):
            skipped += 1
            print(f"[{i}/{len(items)}] 跳过: {data_id or file_name} state={state}")
            continue

        out_name = safe_name_from_data_id(data_id, file_name) + ".zip"
        out_path = output_dir / out_name

        resp = requests.get(zip_url, timeout=300)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        success += 1
        print(f"[{i}/{len(items)}] 下载完成: {out_path}")

    return success, skipped


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


def copy_prepared_files(result: dict[str, Any], extracted_dir: Path, prepared_dir: Path) -> tuple[int, int]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    items = get_extract_items(result)
    kept = 0
    skipped = 0

    for item in items:
        state = str(item.get("state") or "").strip().lower()
        if state != "done":
            skipped += 1
            continue

        data_id = str(item.get("data_id") or "")
        file_name = str(item.get("file_name") or "")
        zip_stem = safe_name_from_data_id(data_id, file_name)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=DEFAULT_TOKEN, help="MinerU API token")
    parser.add_argument("--batch-id", type=str, default=DEFAULT_BATCH_ID)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "processed_pdf")
    parser.add_argument("--extracted-dir", type=Path, default=Path(__file__).parent / "processed_pdf" / "_extracted")
    parser.add_argument("--prepared-dir", type=Path, default=Path(__file__).parent / "processed_txt")
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).parent / "finish_file")
    parser.add_argument("--extract-zip", action="store_true", help="下载后自动解压 ZIP")
    parser.add_argument("--prepare-files", action="store_true", help="整理有效文本文件到 prepared-dir")
    args = parser.parse_args()

    token = (args.token or "").strip()
    if not token:
        raise RuntimeError("请通过 --token 提供 token，或在 DEFAULT_TOKEN 填写")

    batch_id = args.batch_id.strip()
    result = query_batch(token, batch_id)

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = args.result_dir / f"mineru_batch_{batch_id}_{ts}.json"
    save_json(json_path, result)
    print(f"查询结果已保存: {json_path}")

    ok, skipped = download_full_zips(result, args.output_dir)
    print(f"ZIP 下载完成：成功 {ok} 个，跳过 {skipped} 个")

    if args.extract_zip:
        n = extract_zip_files(args.output_dir, args.extracted_dir)
        print(f"ZIP 解压完成：{n} 个")

    if args.prepare_files:
        if not args.extract_zip:
            print("提示：未启用 --extract-zip，将直接基于现有解压目录整理。")
        kept, skipped_items = copy_prepared_files(result, args.extracted_dir, args.prepared_dir)
        print(f"有效文件整理完成：保留 {kept} 个，跳过 {skipped_items} 个条目")


if __name__ == "__main__":
    main()

