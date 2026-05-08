"""
Industry-style automatic evaluation for base vs DPO answers.

This script is intentionally standalone and does not modify existing training
or generation files. It can evaluate the blind_compare.jsonl format produced by
test_dpo_model.py plus blind_compare_key.jsonl.

Core metrics:
  - Semantic relevance: SBERT-style cosine(question, answer)
  - Pair semantic similarity: SBERT-style cosine(base_answer, dpo_answer)
  - Reference semantic similarity: SBERT-style cosine(reference, answer), if references exist
  - BERTScore: contextual similarity against reference, if bert_score and references exist
  - Factual consistency:
      * NLI against evidence/reference, if references exist
      * Optional LLM-as-judge when no reference/evidence is available

Examples:
python eval_standard_metrics.py ^
  --blind_file ./test/blind_compare.jsonl ^
  --key_file ./test/blind_compare_key.jsonl ^
  --output_dir ./test/standard_metrics

python eval_standard_metrics.py ^
  --blind_file ./test/blind_compare.jsonl ^
  --key_file ./test/blind_compare_key.jsonl ^
  --reference_file ./test/reference_answers.jsonl ^
  --run_bertscore ^
  --factual_method nli ^
  --output_dir ./test/standard_metrics_with_ref

python eval_standard_metrics.py ^
  --blind_file ./test/blind_compare.jsonl ^
  --key_file ./test/blind_compare_key.jsonl ^
  --factual_method llm ^
  --judge_model qwen-plus ^
  --api_base https://dashscope.aliyuncs.com/compatible-mode/v1 ^
  --output_dir ./test/standard_metrics_llm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MODEL_NAMES = ("base", "dpo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base and DPO answers with standard automatic metrics.")
    parser.add_argument("--blind_file", type=str, required=True, help="blind_compare.jsonl with id/question/A/B")
    parser.add_argument("--key_file", type=str, required=True, help="blind_compare_key.jsonl with id/A/B -> base/dpo")
    parser.add_argument(
        "--reference_file",
        type=str,
        default=None,
        help=(
            "Optional json/jsonl reference/evidence file. Rows may contain id or question plus "
            "reference/reference_answer/answer/context/evidence."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="standard_metrics", help="Directory for details and summary")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--nli_model", type=str, default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--device", type=str, default=None, help="sentence-transformers/transformers device, e.g. cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--run_bertscore", action="store_true", help="Run BERTScore when references are available")
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="microsoft/deberta-xlarge-mnli",
        help="BERTScore model. For Chinese-only local use, try bert-base-chinese.",
    )
    parser.add_argument(
        "--factual_method",
        choices=["auto", "none", "nli", "llm"],
        default="auto",
        help="auto uses NLI when references exist, otherwise skips factual consistency.",
    )
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str, default=None, help="Fallbacks to OPENAI_API_KEY or DASHSCOPE_API_KEY")
    parser.add_argument("--judge_model", type=str, default=None, help="Required for --factual_method llm")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=260)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return list(read_jsonl(path))
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        rows = obj.get("data") or obj.get("rows") or obj.get("items")
        if isinstance(rows, list):
            return [x for x in rows if isinstance(x, dict)]
        return [obj]
    return []


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def load_key_map(path: Path) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in read_jsonl(path):
        try:
            idx = int(row.get("id"))
        except Exception:
            continue
        a = normalize_text(row.get("A")).lower()
        b = normalize_text(row.get("B")).lower()
        if a in MODEL_NAMES and b in MODEL_NAMES:
            out[idx] = {"A": a, "B": b}
    return out


def pick_reference(row: Dict[str, Any]) -> str:
    for field in ("reference", "reference_answer", "gold", "answer", "context", "evidence"):
        val = normalize_text(row.get(field))
        if val:
            return val
    return ""


def load_references(path: Optional[str]) -> Tuple[Dict[int, str], Dict[str, str]]:
    if not path:
        return {}, {}
    rows = read_json_or_jsonl(Path(path))
    by_id: Dict[int, str] = {}
    by_question: Dict[str, str] = {}
    for row in rows:
        ref = pick_reference(row)
        if not ref:
            continue
        try:
            idx = int(row.get("id"))
            by_id[idx] = ref
        except Exception:
            pass
        q = normalize_text(row.get("question") or row.get("query") or row.get("prompt"))
        if q:
            by_question[q] = ref
    return by_id, by_question


def load_eval_rows(blind_file: Path, key_file: Path, reference_file: Optional[str], max_samples: int) -> List[Dict[str, Any]]:
    key_map = load_key_map(key_file)
    refs_by_id, refs_by_question = load_references(reference_file)
    rows: List[Dict[str, Any]] = []

    for raw in read_jsonl(blind_file):
        try:
            idx = int(raw.get("id"))
        except Exception:
            continue
        key = key_map.get(idx)
        if not key:
            continue
        question = normalize_text(raw.get("question"))
        answer_a = normalize_text(raw.get("A"))
        answer_b = normalize_text(raw.get("B"))
        if not question or not answer_a or not answer_b:
            continue
        answers = {key["A"]: answer_a, key["B"]: answer_b}
        if "base" not in answers or "dpo" not in answers:
            continue
        rows.append(
            {
                "id": idx,
                "question": question,
                "base_answer": answers["base"],
                "dpo_answer": answers["dpo"],
                "reference": refs_by_id.get(idx) or refs_by_question.get(question) or "",
            }
        )
        if max_samples and max_samples > 0 and len(rows) >= max_samples:
            break
    return rows


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def load_embedder(model_name: str, device: Optional[str]):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install sentence-transformers") from exc
    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


def encode_texts(embedder: Any, texts: List[str], batch_size: int) -> List[List[float]]:
    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return vectors.tolist()


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s+|\n+", text)
    claims = [normalize_text(p) for p in parts if len(normalize_text(p)) >= 8]
    return claims[:24]


def label_score(output: Any, label_substr: str) -> float:
    if isinstance(output, dict):
        output = [output]
    best = 0.0
    for item in output or []:
        label = str(item.get("label", "")).lower()
        if label_substr in label:
            best = max(best, float(item.get("score", 0.0)))
    return best


def run_nli_consistency(rows: List[Dict[str, Any]], model_name: str, device: Optional[str]) -> None:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install transformers torch") from exc

    pipe_kwargs: Dict[str, Any] = {"model": model_name, "top_k": None, "truncation": True}
    if device:
        pipe_kwargs["device"] = 0 if device.startswith("cuda") else -1
    clf = pipeline("text-classification", **pipe_kwargs)

    for row in rows:
        evidence = row.get("reference", "")
        if not evidence:
            continue
        for model_name_key in MODEL_NAMES:
            claims = sentence_split(row[f"{model_name_key}_answer"])
            if not claims:
                row[f"{model_name_key}_nli_entailment"] = None
                row[f"{model_name_key}_nli_contradiction"] = None
                row[f"{model_name_key}_fact_consistency"] = None
                continue
            entail_scores = []
            contra_scores = []
            for claim in claims:
                output = clf({"text": evidence, "text_pair": claim})
                entail_scores.append(label_score(output, "entail"))
                contra_scores.append(label_score(output, "contrad"))
            entail = statistics.mean(entail_scores)
            contradiction = statistics.mean(contra_scores)
            row[f"{model_name_key}_nli_entailment"] = entail
            row[f"{model_name_key}_nli_contradiction"] = contradiction
            row[f"{model_name_key}_fact_consistency"] = entail * (1.0 - contradiction)


LLM_FACT_SYSTEM = (
    "你是严格的医疗事实一致性评估员。请只根据医学常识、题目和可选参考证据，"
    "判断回答是否存在事实错误、无依据夸大或危险建议。"
)


LLM_FACT_TEMPLATE = """请评估回答的事实一致性，输出严格 JSON。

题目：
{question}

参考证据（可能为空）：
{reference}

回答：
{answer}

评分标准：
5 = 几乎完全事实一致，且没有危险建议
4 = 基本一致，仅有轻微不严谨
3 = 有部分未经证实或含混表述，但主要结论可接受
2 = 存在明显事实问题或可能误导
1 = 关键事实错误、危险建议或严重幻觉

输出格式：
{{"score": 1-5, "reason": "不超过40字"}}

仅输出 JSON。"""


def parse_json_from_text(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def call_chat_api(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Missing dependency for LLM judge: pip install requests") from exc

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def run_llm_fact_consistency(rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    api_key = normalize_text(args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"))
    if not api_key:
        raise ValueError("LLM factual evaluation needs --api_key or OPENAI_API_KEY/DASHSCOPE_API_KEY.")
    if not args.judge_model:
        raise ValueError("LLM factual evaluation needs --judge_model.")

    for i, row in enumerate(rows, 1):
        for model_name_key in MODEL_NAMES:
            user_prompt = LLM_FACT_TEMPLATE.format(
                question=row["question"],
                reference=row.get("reference", ""),
                answer=row[f"{model_name_key}_answer"],
            )
            raw = call_chat_api(
                api_base=args.api_base,
                api_key=api_key,
                model=args.judge_model,
                messages=[
                    {"role": "system", "content": LLM_FACT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            parsed = parse_json_from_text(raw)
            try:
                score = float(parsed.get("score"))
            except Exception:
                score = None
            row[f"{model_name_key}_llm_fact_score"] = score
            row[f"{model_name_key}_llm_fact_reason"] = normalize_text(parsed.get("reason") or raw)[:240]
            row[f"{model_name_key}_fact_consistency"] = score / 5.0 if score is not None else None
            if args.sleep > 0:
                time.sleep(args.sleep)
        if i % 5 == 0 or i == len(rows):
            print(f"[LLM factual] {i}/{len(rows)} done")


def run_bertscore(rows: List[Dict[str, Any]], model_type: str, device: Optional[str]) -> None:
    ref_rows = [row for row in rows if row.get("reference")]
    if not ref_rows:
        return
    try:
        from bert_score import score
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install bert-score") from exc

    refs = [row["reference"] for row in ref_rows]
    for model_name_key in MODEL_NAMES:
        candidates = [row[f"{model_name_key}_answer"] for row in ref_rows]
        kwargs: Dict[str, Any] = {"model_type": model_type, "verbose": True, "rescale_with_baseline": False}
        if device:
            kwargs["device"] = device
        precision, recall, f1 = score(candidates, refs, **kwargs)
        for row, p, r, f in zip(ref_rows, precision.tolist(), recall.tolist(), f1.tolist()):
            row[f"{model_name_key}_bertscore_p"] = p
            row[f"{model_name_key}_bertscore_r"] = r
            row[f"{model_name_key}_bertscore_f1"] = f


def add_embedding_metrics(rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    embedder = load_embedder(args.embedding_model, args.device)
    texts: List[str] = []
    for row in rows:
        texts.extend([row["question"], row["base_answer"], row["dpo_answer"]])
        if row.get("reference"):
            texts.append(row["reference"])

    unique_texts = list(dict.fromkeys(texts))
    vectors = encode_texts(embedder, unique_texts, args.batch_size)
    vector_map = dict(zip(unique_texts, vectors))

    for row in rows:
        qv = vector_map[row["question"]]
        base_v = vector_map[row["base_answer"]]
        dpo_v = vector_map[row["dpo_answer"]]
        row["base_semantic_relevance"] = cosine(qv, base_v)
        row["dpo_semantic_relevance"] = cosine(qv, dpo_v)
        row["base_dpo_semantic_similarity"] = cosine(base_v, dpo_v)
        if row.get("reference"):
            rv = vector_map[row["reference"]]
            row["base_reference_similarity"] = cosine(rv, base_v)
            row["dpo_reference_similarity"] = cosine(rv, dpo_v)


def mean_or_none(values: List[Any]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not nums:
        return None
    return statistics.mean(nums)


def summarize(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    metrics = [
        "semantic_relevance",
        "reference_similarity",
        "bertscore_p",
        "bertscore_r",
        "bertscore_f1",
        "nli_entailment",
        "nli_contradiction",
        "llm_fact_score",
        "fact_consistency",
    ]
    summary: Dict[str, Any] = {
        "total_rows": len(rows),
        "rows_with_reference": sum(1 for r in rows if r.get("reference")),
        "embedding_model": args.embedding_model,
        "factual_method": args.factual_method,
        "nli_model": args.nli_model if args.factual_method in {"auto", "nli"} else None,
        "judge_model": args.judge_model if args.factual_method == "llm" else None,
    }
    for metric in metrics:
        for model_name_key in MODEL_NAMES:
            key = f"{model_name_key}_{metric}"
            val = mean_or_none([row.get(key) for row in rows])
            if val is not None:
                summary[f"avg_{key}"] = val
        base_key = f"base_{metric}"
        dpo_key = f"dpo_{metric}"
        base_avg = summary.get(f"avg_{base_key}")
        dpo_avg = summary.get(f"avg_{dpo_key}")
        if base_avg is not None and dpo_avg is not None:
            summary[f"delta_dpo_minus_base_{metric}"] = dpo_avg - base_avg

    pair_avg = mean_or_none([row.get("base_dpo_semantic_similarity") for row in rows])
    if pair_avg is not None:
        summary["avg_base_dpo_semantic_similarity"] = pair_avg
    return summary


def write_outputs(rows: List[Dict[str, Any]], summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details_jsonl = output_dir / "details.jsonl"
    details_csv = output_dir / "details.csv"
    summary_json = output_dir / "summary.json"

    with details_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with details_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    rows = load_eval_rows(Path(args.blind_file), Path(args.key_file), args.reference_file, args.max_samples)
    if not rows:
        raise ValueError("No valid rows loaded. Check --blind_file and --key_file.")

    print(f"Loaded {len(rows)} rows; references: {sum(1 for r in rows if r.get('reference'))}")
    add_embedding_metrics(rows, args)

    has_reference = any(row.get("reference") for row in rows)
    if args.run_bertscore and has_reference:
        run_bertscore(rows, args.bertscore_model, args.device)

    factual_method = args.factual_method
    if factual_method == "auto":
        factual_method = "nli" if has_reference else "none"
    if factual_method == "nli":
        if not has_reference:
            print("Skip NLI factual consistency: no reference/evidence provided.")
        else:
            run_nli_consistency(rows, args.nli_model, args.device)
    elif factual_method == "llm":
        run_llm_fact_consistency(rows, args)

    summary = summarize(rows, args)
    write_outputs(rows, summary, Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved details and summary to: {args.output_dir}")


if __name__ == "__main__":
    main()
