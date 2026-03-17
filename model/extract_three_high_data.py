import csv
import json
import re
import argparse
from pathlib import Path


ROOT = Path("SFT_data")
OUT_ROOT = ROOT / "three_high_subset"


KEYWORDS = [
    "高血压",
    "血压高",
    "高血糖",
    "血糖高",
    "糖尿病",
    "高血脂",
    "血脂高",
    "高脂血症",
    "血脂异常",
    "甘油三酯高",
    "胆固醇高",
    "三高",
]
PATTERN = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.IGNORECASE)


def is_related(text: str) -> bool:
    return bool(PATTERN.search(text or ""))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def filter_huatuo_file(src: Path, dst: Path) -> int:
    keep = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            questions = item.get("questions", [])
            q_text = " ".join(
                q if isinstance(q, str) else " ".join(str(x) for x in q)
                for q in questions
            )
            if is_related(q_text):
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                keep += 1
    return keep


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def write_csv_rows(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_cmedqa2(src_dir: Path, dst_dir: Path) -> dict:
    question_fields, question_rows = read_csv_rows(src_dir / "question.csv")
    answer_fields, answer_rows = read_csv_rows(src_dir / "answer.csv")

    # Strict mode: only question text decides whether the sample is "three-high related".
    selected_qids = {
        row["question_id"]
        for row in question_rows
        if is_related(row.get("content", ""))
    }

    answer_rows_by_qid = {}
    for row in answer_rows:
        qid = row["question_id"]
        answer_rows_by_qid.setdefault(qid, []).append(row)
    selected_questions = [row for row in question_rows if row["question_id"] in selected_qids]

    selected_answers = []
    selected_ans_ids = set()
    for qid in selected_qids:
        for row in answer_rows_by_qid.get(qid, []):
            selected_answers.append(row)
            selected_ans_ids.add(row["ans_id"])

    candidate_summary = {}
    candidate_files = ["train_candidates.txt", "dev_candidates.txt", "test_candidates.txt"]
    for name in candidate_files:
        src = src_dir / name
        dst = dst_dir / name
        kept = []
        with src.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            for row in reader:
                qid = row.get("question_id", "")
                pos = row.get("pos_ans_id")
                neg = row.get("neg_ans_id")
                # Keep candidates only when both referenced answers are available in subset.
                if qid in selected_qids and pos in selected_ans_ids and neg in selected_ans_ids:
                    kept.append(row)

        with dst.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        candidate_summary[name] = len(kept)

    selected_answers.sort(key=lambda x: int(x["ans_id"]))
    write_csv_rows(dst_dir / "question.csv", question_fields, selected_questions)
    write_csv_rows(dst_dir / "answer.csv", answer_fields, selected_answers)

    return {
        "question_count": len(selected_questions),
        "answer_count": len(selected_answers),
        **candidate_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract three-high related subset from SFT_data.")
    parser.add_argument(
        "--strict-question-only",
        action="store_true",
        default=True,
        help="Reserved flag. Current behavior is always strict question-only matching.",
    )
    parser.parse_args()

    ensure_dir(OUT_ROOT)

    huatuo_in = ROOT / "huatuo_qa"
    huatuo_out = OUT_ROOT / "huatuo_qa"
    ensure_dir(huatuo_out)
    huatuo_stats = {}
    for name in ["train_datasets.jsonl", "validation_datasets.jsonl", "test_datasets.jsonl"]:
        kept = filter_huatuo_file(huatuo_in / name, huatuo_out / name)
        huatuo_stats[name] = kept

    cmed_in = ROOT / "cMedQA2"
    cmed_out = OUT_ROOT / "cMedQA2"
    ensure_dir(cmed_out)
    cmed_stats = filter_cmedqa2(cmed_in, cmed_out)

    summary = {
        "keywords": KEYWORDS,
        "huatuo_qa": huatuo_stats,
        "cMedQA2": cmed_stats,
    }
    with (OUT_ROOT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
