"""
测试脚本：加载 outputs/qwen25_rag_lora 中保存的 LoRA 微调模型并进行推理。
支持单条问答、批量测试样例、以及交互式对话三种模式。
"""

import argparse
import json
import sys
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_SYSTEM_PROMPT = "你是一个严谨的医疗助手，请基于提供的检索资料回答问题。若资料不足，明确说明不确定。"

# 内置测试样例（无检索上下文 / 有检索上下文各若干条）
BUILTIN_SAMPLES = [
    {
        "question": "高血压患者的正常血压目标值是多少？",
        "context": "",
    },
    {
        "question": "二甲双胍的主要副作用有哪些？",
        "context": (
            "二甲双胍是2型糖尿病的一线用药，常见不良反应包括胃肠道反应（恶心、腹泻、腹痛），"
            "长期使用可能导致维生素B12吸收减少。肾功能不全（eGFR<30）时禁用，以防乳酸酸中毒。"
        ),
    },
    {
        "question": "什么是高脂血症？应如何干预？",
        "context": (
            "高脂血症是指血浆中胆固醇和/或甘油三酯水平升高，或高密度脂蛋白胆固醇水平降低。"
            "干预措施包括：生活方式改变（低脂饮食、增加运动、戒烟）、他汀类药物（如阿托伐他汀、瑞舒伐他汀）、"
            "以及贝特类或烟酸类药物用于高甘油三酯血症。"
        ),
    },
    {
        "question": "患者，男，60岁，血压160/100mmHg，已服用氨氯地平5mg一个月，血压未达标，下一步如何处理？",
        "context": (
            "单药治疗血压未达标时，可考虑以下方案：①增加当前药物剂量；"
            "②联合另一类降压药（如ACEI/ARB + CCB，或CCB + 利尿剂）；"
            "③换用另一类降压药。指南推荐对大多数患者采用联合治疗以提高达标率。"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 Qwen2.5 RAG-LoRA 微调模型")
    parser.add_argument("--base_model", type=str, default="model", help="基础模型路径")
    parser.add_argument("--lora_path", type=str, default="outputs/qwen25_rag_lora", help="LoRA adapter 路径")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_4bit", action="store_true", help="以4bit量化加载（节省显存）")
    parser.add_argument("--bf16", action="store_true", help="使用 bf16 精度")
    parser.add_argument(
        "--mode",
        type=str,
        default="samples",
        choices=["samples", "interactive", "single"],
        help="运行模式：samples=内置测试样例, interactive=交互式, single=单条（需提供 --question）",
    )
    parser.add_argument("--question", type=str, default=None, help="single 模式下的问题")
    parser.add_argument("--context", type=str, default=None, help="single 模式下的检索上下文（可选）")
    parser.add_argument("--test_file", type=str, default=None, help="从 jsonl 文件批量测试（每行含 question/context/answer）")
    parser.add_argument("--output_file", type=str, default=None, help="将批量测试结果写入此文件")
    return parser.parse_args()


def load_model(args):
    print(f"[1/2] 加载 tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 推理时左填充

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    print(f"[2/2] 加载基础模型并合并 LoRA adapter: {args.lora_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=dtype if not args.use_4bit else None,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    print("模型加载完毕。\n")
    return model, tokenizer


def build_prompt(tokenizer, system_prompt: str, question: str, context: Optional[str]) -> str:
    question = (question or "").strip()
    context = (context or "").strip()
    user_text = f"问题：{question}\n\n检索资料：\n{context}" if context else f"问题：{question}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, args) -> str:
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
    # 只解码新生成的 token
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_samples(model, tokenizer, args):
    print("=" * 60)
    print("内置测试样例")
    print("=" * 60)
    for i, sample in enumerate(BUILTIN_SAMPLES, 1):
        question = sample["question"]
        context = sample.get("context", "")
        prompt = build_prompt(tokenizer, args.system_prompt, question, context)
        answer = generate(model, tokenizer, prompt, args)

        print(f"\n【样例 {i}】")
        print(f"问题: {question}")
        if context:
            preview = context[:80] + "..." if len(context) > 80 else context
            print(f"上下文: {preview}")
        print(f"回答: {answer}")
        print("-" * 60)


def run_interactive(model, tokenizer, args):
    print("=" * 60)
    print("交互式测试（输入 'quit' 或 'exit' 退出，'clear' 清空上下文）")
    print("=" * 60)
    while True:
        try:
            question = input("\n请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if question.lower() in ("quit", "exit"):
            print("退出。")
            break
        if not question:
            continue

        context = input("检索上下文（可直接回车跳过）: ").strip()
        prompt = build_prompt(tokenizer, args.system_prompt, question, context)
        print("\n生成中...", flush=True)
        answer = generate(model, tokenizer, prompt, args)
        print(f"\n回答:\n{answer}")


def run_single(model, tokenizer, args):
    if not args.question:
        print("single 模式需要提供 --question 参数。", file=sys.stderr)
        sys.exit(1)
    prompt = build_prompt(tokenizer, args.system_prompt, args.question, args.context)
    answer = generate(model, tokenizer, prompt, args)
    print(f"问题: {args.question}")
    if args.context:
        print(f"上下文: {args.context}")
    print(f"\n回答:\n{answer}")


def run_file(model, tokenizer, args):
    results = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    print(f"共 {len(lines)} 条测试数据，开始推理...")
    for i, item in enumerate(lines, 1):
        question = str(item.get("question", ""))
        context = str(item.get("context", "") or "")
        reference = str(item.get("answer", "") or "")
        prompt = build_prompt(tokenizer, args.system_prompt, question, context)
        prediction = generate(model, tokenizer, prompt, args)
        results.append({"question": question, "context": context, "reference": reference, "prediction": prediction})
        print(f"[{i}/{len(lines)}] {question[:40]}... => {prediction[:60]}...")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n结果已写入: {args.output_file}")
    else:
        print("\n--- 推理结果 ---")
        for r in results:
            print(json.dumps(r, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    model, tokenizer = load_model(args)

    if args.test_file:
        run_file(model, tokenizer, args)
    elif args.mode == "samples":
        run_samples(model, tokenizer, args)
    elif args.mode == "interactive":
        run_interactive(model, tokenizer, args)
    elif args.mode == "single":
        run_single(model, tokenizer, args)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 使用示例
# ---------------------------------------------------------------------------
# 1. 内置测试样例（默认）
#    python test_qwen25_rag_lora.py --bf16
#
# 2. 显存不足时使用 4bit 量化
#    python test_qwen25_rag_lora.py --use_4bit --bf16
#
# 3. 交互式问答
#    python test_qwen25_rag_lora.py --mode interactive --bf16
#
# 4. 单条测试（无上下文）
#    python test_qwen25_rag_lora.py --mode single \
#        --question "高血压有哪些并发症？" --bf16
#
# 5. 单条测试（带检索上下文）
#    python test_qwen25_rag_lora.py --mode single \
#        --question "如何调整降压方案？" \
#        --context "单药未达标时可联合用药..." --bf16
#
# 6. 批量测试 jsonl 文件并保存结果
#    python test_qwen25_rag_lora.py \
#        --test_file SFT_data/three_high_subset/merged_three_high_qa.dedup.jsonl \
#        --output_file outputs/test_predictions.jsonl --bf16
