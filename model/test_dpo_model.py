"""
测试 DPO 模型：
1) preference 模式：在 DPO 数据上计算 chosen 胜率（基于条件对数概率）
2) single / interactive 模式：常规生成测试

python test_dpo_model.py \
  --mode blind_compare \
  --bf16 \
  --base_model /hy-tmp \
  --lora_path ./outputs/qwen25_dpo_lora \
  --question_file ./non_train_questions.txt \
  --sample_size 120 \
  --blind_output_file ./blind_compare.jsonl \
  --blind_key_file ./blind_compare_key.jsonl
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Optional

from qwen_service.prompts import DEFAULT_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Qwen DPO model")
    parser.add_argument("--base_model", type=str, default="qwen2.5-1.7B", help="基础模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="可选 LoRA adapter 路径")

    parser.add_argument(
        "--mode",
        type=str,
        default="preference",
        choices=["preference", "single", "interactive", "blind_compare"],
    )
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)

    parser.add_argument("--dpo_file", type=str, default="SFT_data/three_high_subset/generated_dpo.jsonl")
    parser.add_argument("--prompt_field", type=str, default="prompt")
    parser.add_argument("--chosen_field", type=str, default="chosen")
    parser.add_argument("--rejected_field", type=str, default="rejected")
    parser.add_argument("--max_samples", type=int, default=0, help="0=all")
    parser.add_argument("--output_file", type=str, default=None, help="保存偏好评估明细")

    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--context", type=str, default=None)
    parser.add_argument("--question_file", type=str, default=None, help="盲测问题文件（txt/json/jsonl）")
    parser.add_argument("--sample_size", type=int, default=20, help="盲测抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--exclude_train_file",
        type=str,
        default="SFT_data/three_high_subset/generated_dpo.jsonl",
        help="用于剔除训练集中问题的文件；传空字符串可关闭",
    )
    parser.add_argument("--blind_output_file", type=str, default="blind_compare.jsonl", help="盲测输出文件（A/B）")
    parser.add_argument("--blind_key_file", type=str, default="blind_compare_key.jsonl", help="盲测映射文件")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def load_model(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[1/2] 加载 tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    print(f"[2/2] 加载基础模型{(' + LoRA: ' + args.lora_path) if args.lora_path else ''}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=dtype if not args.use_4bit else None,
        device_map="auto",
    )

    if args.lora_path:
        model = PeftModel.from_pretrained(base_model, args.lora_path)
    else:
        model = base_model

    model.eval()
    print("模型加载完毕。\n")
    return model, tokenizer


def build_prompt(tokenizer, system_prompt: str, question: str, context: Optional[str]) -> str:
    q = (question or "").strip()
    c = (context or "").strip()
    user_text = f"问题：{q}\n\n检索资料：\n{c}" if c else f"问题：{q}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, prompt: str, args) -> str:
    import torch

    @torch.inference_mode()
    def _inner():
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return _inner()


def _model_device_summary(model) -> str:
    hfm = getattr(model, "hf_device_map", None)
    if not hfm:
        try:
            return str(model.device)
        except Exception:
            return "unknown"
    devices = sorted({str(v) for v in hfm.values()})
    return ",".join(devices)


def continuation_logprob(model, tokenizer, prompt: str, continuation: str) -> float:
    import torch

    @torch.inference_mode()
    def _inner() -> float:
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        cont_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"]

        input_ids = torch.cat([prompt_ids, cont_ids], dim=1).to(model.device)
        attn = torch.ones_like(input_ids, device=model.device)

        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]
        target = input_ids[:, 1:]

        prompt_len = prompt_ids.shape[1]
        cont_len = cont_ids.shape[1]
        if cont_len == 0:
            return float("-inf")

        # Target positions corresponding to continuation tokens start at prompt_len-1
        start = max(prompt_len - 1, 0)
        end = start + cont_len

        log_probs = torch.log_softmax(logits[:, start:end, :], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target[:, start:end].unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.sum().item())

    return _inner()


def run_preference_eval(model, tokenizer, args):
    rows = []
    with open(args.dpo_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    if not rows:
        print("没有可评估数据。")
        return

    n = 0
    wins = 0
    margins = []
    details = []

    print(f"开始偏好评估，共 {len(rows)} 条...")
    for i, item in enumerate(rows, 1):
        prompt = str(item.get(args.prompt_field, "") or "").strip()
        chosen = str(item.get(args.chosen_field, "") or "").strip()
        rejected = str(item.get(args.rejected_field, "") or "").strip()
        if not prompt or not chosen or not rejected:
            continue

        lp_chosen = continuation_logprob(model, tokenizer, prompt, chosen)
        lp_rejected = continuation_logprob(model, tokenizer, prompt, rejected)
        margin = lp_chosen - lp_rejected

        n += 1
        if margin > 0:
            wins += 1
        margins.append(margin)

        details.append(
            {
                "index": i,
                "margin": margin,
                "chosen_logprob": lp_chosen,
                "rejected_logprob": lp_rejected,
            }
        )

        if i % 10 == 0:
            print(f"[{i}/{len(rows)}] 当前胜率={wins / max(n, 1):.4f}")

    if n == 0:
        print("有效样本数为 0。")
        return

    acc = wins / n
    avg_margin = sum(margins) / n
    std = math.sqrt(sum((m - avg_margin) ** 2 for m in margins) / n)

    print("\n=== DPO 偏好评估结果 ===")
    print(f"样本数: {n}")
    print(f"chosen 胜率: {acc:.4f}")
    print(f"平均 margin: {avg_margin:.4f}")
    print(f"margin 标准差: {std:.4f}")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"评估明细已写入: {args.output_file}")


def run_single(model, tokenizer, args):
    if not args.question:
        print("single 模式需要 --question", file=sys.stderr)
        sys.exit(1)
    prompt = build_prompt(tokenizer, args.system_prompt, args.question, args.context)
    answer = generate(model, tokenizer, prompt, args)
    print(f"问题: {args.question}")
    if args.context:
        print(f"上下文: {args.context}")
    print(f"\n回答:\n{answer}")


def run_interactive(model, tokenizer, args):
    print("交互式测试（输入 quit 或 exit 退出）")
    while True:
        try:
            question = input("\n请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            return
        if question.lower() in {"quit", "exit"}:
            print("退出。")
            return
        if not question:
            continue
        context = input("检索上下文（可回车跳过）: ").strip()
        prompt = build_prompt(tokenizer, args.system_prompt, question, context)
        print("生成中...")
        print(generate(model, tokenizer, prompt, args))


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def _extract_question_from_prompt(prompt: str) -> str:
    p = (prompt or "").strip()
    if not p:
        return ""
    marker = "问题："
    idx = p.rfind(marker)
    if idx >= 0:
        p = p[idx + len(marker) :]
    sep = "检索资料："
    if sep in p:
        p = p.split(sep, 1)[0]
    return _norm_text(p)


def load_questions_for_blind_eval(args) -> list[str]:
    if not args.question_file:
        raise ValueError("blind_compare 模式需要 --question_file")
    if not os.path.exists(args.question_file):
        raise FileNotFoundError(f"问题文件不存在: {args.question_file}")

    questions: list[str] = []
    path = args.question_file
    lower = path.lower()

    if lower.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                q = _norm_text(line)
                if q:
                    questions.append(q)
    elif lower.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    q = _norm_text(item)
                else:
                    q = _norm_text(str(item.get("question", "") or item.get("query", "") or item.get("prompt", "")))
                if q:
                    if "prompt" in item and not item.get("question"):
                        q = _extract_question_from_prompt(str(item.get("prompt", "")))
                    if q:
                        questions.append(q)
    elif lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    q = _norm_text(item)
                else:
                    q = _norm_text(str(item.get("question", "") or item.get("query", "") or item.get("prompt", "")))
                    if "prompt" in item and not item.get("question"):
                        q = _extract_question_from_prompt(str(item.get("prompt", "")))
                if q:
                    questions.append(q)
    else:
        raise ValueError("question_file 仅支持 .txt / .jsonl / .json")

    # 去重，保持顺序
    dedup = []
    seen = set()
    for q in questions:
        key = _norm_text(q)
        if key and key not in seen:
            seen.add(key)
            dedup.append(key)
    return dedup


def load_train_questions(args) -> set[str]:
    path = (args.exclude_train_file or "").strip()
    if not path:
        return set()
    if not os.path.exists(path):
        print(f"警告：exclude_train_file 不存在，跳过训练集过滤: {path}")
        return set()

    train_qs = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = _norm_text(str(item.get("question", "") or ""))
            if not q:
                q = _extract_question_from_prompt(str(item.get("prompt", "") or ""))
            if q:
                train_qs.add(q)
    return train_qs


def run_blind_compare(args):
    if not args.lora_path:
        print("blind_compare 模式需要提供 --lora_path", file=sys.stderr)
        sys.exit(1)

    print("[1/4] 读取问题池并过滤训练集...")
    question_pool = load_questions_for_blind_eval(args)
    train_qs = load_train_questions(args)
    if train_qs:
        question_pool = [q for q in question_pool if _norm_text(q) not in train_qs]
    if not question_pool:
        print("过滤后无可用问题，请检查 question_file / exclude_train_file", file=sys.stderr)
        sys.exit(1)

    rnd = random.Random(args.seed)
    if args.sample_size and args.sample_size > 0:
        sample_size = min(args.sample_size, len(question_pool))
        sampled_questions = rnd.sample(question_pool, sample_size)
    else:
        sampled_questions = question_pool

    # base 模型：先完整跑一遍，避免和 DPO 同时驻留导致 offload 变慢
    base_args = argparse.Namespace(**vars(args))
    base_args.lora_path = None
    print("[2/4] 加载 base 模型...")
    base_model, tokenizer = load_model(base_args)
    print(f"base 设备映射: {_model_device_summary(base_model)}")

    print(f"[3/5] 生成 base 答案，共 {len(sampled_questions)} 题...")
    base_answers = {}
    for i, q in enumerate(sampled_questions, 1):
        prompt = build_prompt(tokenizer, args.system_prompt, q, None)
        base_answers[q] = generate(base_model, tokenizer, prompt, args)
        if i % 5 == 0 or i == len(sampled_questions):
            print(f"[base {i}/{len(sampled_questions)}] 完成")

    # 释放 base，给 DPO 模型腾显存
    del base_model
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print("[4/5] 加载 DPO 模型（base + LoRA）...")
    dpo_model, _ = load_model(args)
    print(f"dpo 设备映射: {_model_device_summary(dpo_model)}")

    print(f"[5/5] 生成 DPO 答案并写出盲测文件，共 {len(sampled_questions)} 题...")
    blind_rows = []
    key_rows = []

    for i, q in enumerate(sampled_questions, 1):
        prompt = build_prompt(tokenizer, args.system_prompt, q, None)
        base_ans = base_answers[q]
        dpo_ans = generate(dpo_model, tokenizer, prompt, args)

        if rnd.random() < 0.5:
            a_ans, b_ans = base_ans, dpo_ans
            key = {"A": "base", "B": "dpo"}
        else:
            a_ans, b_ans = dpo_ans, base_ans
            key = {"A": "dpo", "B": "base"}

        blind_rows.append(
            {
                "id": i,
                "question": q,
                "A": a_ans,
                "B": b_ans,
            }
        )
        key_rows.append({"id": i, **key})

        if i % 5 == 0 or i == len(sampled_questions):
            print(f"[dpo {i}/{len(sampled_questions)}] 完成")

    with open(args.blind_output_file, "w", encoding="utf-8") as f:
        for r in blind_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.blind_key_file, "w", encoding="utf-8") as f:
        for r in key_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== 盲测文件已生成 ===")
    print(f"盲测题目数: {len(sampled_questions)}")
    print(f"盲测文件(A/B): {args.blind_output_file}")
    print(f"映射文件(key): {args.blind_key_file}")
    print("请先人工盲评 blind_output_file，再用 blind_key_file 对照统计。")


def main():
    args = parse_args()
    if args.mode == "blind_compare":
        run_blind_compare(args)
        return

    model, tokenizer = load_model(args)
    if args.mode == "preference":
        run_preference_eval(model, tokenizer, args)
    elif args.mode == "single":
        run_single(model, tokenizer, args)
    elif args.mode == "interactive":
        run_interactive(model, tokenizer, args)


if __name__ == "__main__":
    main()

'''
示例：
python test_dpo_model.py --mode preference --bf16 \\
  --base_model qwen2.5-1.7B --lora_path outputs/qwen25_dpo_lora \\
  --dpo_file SFT_data/three_high_subset/generated_dpo.jsonl --max_samples 200

python test_dpo_model.py --mode single --bf16 \\
  --base_model qwen2.5-1.7B --lora_path outputs/qwen25_dpo_lora \\
  --question "高血糖有什么症状"
'''
