import csv
import json
from pathlib import Path


BASE = Path("SFT_data/three_high_subset")
HUATUO_DIR = BASE / "huatuo_qa"
CMED_DIR = BASE / "cMedQA2"
OUT_FILE = BASE / "merged_three_high_qa.jsonl"


def flatten_questions(questions):
    flat = []
    for q in questions or []:
        if isinstance(q, str):
            flat.append(q.strip())
        elif isinstance(q, list):
            for x in q:
                text = str(x).strip()
                if text:
                    flat.append(text)
    return [x for x in flat if x]


def merge_huatuo(writer) -> int:
    count = 0
    files = ["train_datasets.jsonl", "validation_datasets.jsonl", "test_datasets.jsonl"]
    for name in files:
        path = HUATUO_DIR / name
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                questions = flatten_questions(item.get("questions", []))
                answers = [str(a).strip() for a in item.get("answers", []) if str(a).strip()]
                for q in questions:
                    for a in answers:
                        writer.write(
                            json.dumps(
                                {
                                    "source": "huatuo_qa",
                                    "question": q,
                                    "answer": a,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        count += 1
    return count


def merge_cmed(writer) -> int:
    count = 0
    questions = {}
    with (CMED_DIR / "question.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            questions[row["question_id"]] = row.get("content", "").strip()

    with (CMED_DIR / "answer.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["question_id"]
            q = questions.get(qid, "").strip()
            a = row.get("content", "").strip()
            if not q or not a:
                continue
            writer.write(
                json.dumps(
                    {
                        "source": "cMedQA2",
                        "question": q,
                        "answer": a,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def main():
    total = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        total += merge_huatuo(out)
        total += merge_cmed(out)

    summary = {"output_file": str(OUT_FILE), "total_records": total}
    with (BASE / "merged_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
