import argparse
import inspect
import os
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


DEFAULT_SYSTEM_PROMPT = "你是一个严谨的医疗助手，请基于提供的检索资料回答问题。若资料不足，明确说明不确定。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen2.5-7B-Instruct with RAG-style input.")
    parser.add_argument("--model_path", type=str, default="model", help="Local model path.")
    parser.add_argument(
        "--train_file",
        type=str,
        default="SFT_data/three_high_subset/merged_three_high_qa.dedup.jsonl",
        help="Training jsonl file.",
    )
    parser.add_argument("--val_file", type=str, default=None, help="Optional validation jsonl file.")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen25_rag_lora")
    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument("--context_field", type=str, default="context")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit QLoRA.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_user_text(question: str, context: Optional[str]) -> str:
    question = (question or "").strip()
    context = (context or "").strip()
    if context:
        return f"问题：{question}\n\n检索资料：\n{context}"
    return f"问题：{question}"


def make_preprocess_fn(tokenizer, args):
    def preprocess(example: Dict[str, str]) -> Dict[str, List[int]]:
        question = str(example.get(args.question_field, "") or "")
        answer = str(example.get(args.answer_field, "") or "").strip()
        context = str(example.get(args.context_field, "") or "")

        user_text = build_user_text(question, context)
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + answer + (tokenizer.eos_token or "")

        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )
        tokenized_prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        prompt_len = len(tokenized_prompt["input_ids"])

        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return preprocess


def load_data(args, tokenizer):
    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file
    raw = load_dataset("json", data_files=data_files)
    preprocess = make_preprocess_fn(tokenizer, args)
    tokenized = raw.map(
        preprocess,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )
    return tokenized


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else "auto"),
        device_map="auto",
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    tokenized = load_data(args, tokenizer)

    eval_strategy = "steps" if "validation" in tokenized else "no"
    ta_sig = inspect.signature(TrainingArguments.__init__)
    eval_arg_name = "eval_strategy" if "eval_strategy" in ta_sig.parameters else "evaluation_strategy"

    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "optim": "paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "report_to": "none",
        "seed": args.seed,
        "dataloader_num_workers": 2,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        eval_arg_name: eval_strategy,
    }
    if eval_strategy == "steps":
        training_kwargs["eval_steps"] = args.eval_steps

    training_args = TrainingArguments(**training_kwargs)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized.get("validation"),
        "data_collator": data_collator,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training done. LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

'''
python train_qwen25_rag_lora.py \
  --model_path model \
  --train_file SFT_data/three_high_subset/merged_three_high_qa.dedup.jsonl \
  --output_dir outputs/qwen25_rag_lora \
  --use_4bit --bf16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --max_length 1024
'''