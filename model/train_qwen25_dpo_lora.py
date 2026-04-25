import argparse
import inspect
import os
import random
from typing import Dict, Optional, Any

from qwen_service.prompts import DEFAULT_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO fine-tuning for Qwen with optional LoRA/QLoRA")
    parser.add_argument("--model_path", type=str, default="qwen2.5-1.7B", help="Base model path")
    parser.add_argument(
        "--train_file",
        type=str,
        default="SFT_data/three_high_subset/generated_dpo.jsonl",
        help="DPO train jsonl (prompt/chosen/rejected)",
    )
    parser.add_argument("--val_file", type=str, default=None, help="Optional validation jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen25_dpo_lora")

    parser.add_argument("--prompt_field", type=str, default="prompt")
    parser.add_argument("--chosen_field", type=str, default="chosen")
    parser.add_argument("--rejected_field", type=str, default="rejected")

    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--context_field", type=str, default="context")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)

    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated target modules for LoRA",
    )

    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit QLoRA loading")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_user_text(question: str, context: Optional[str]) -> str:
    q = (question or "").strip()
    c = (context or "").strip()
    if c:
        return f"问题：{q}\\n\\n检索资料：\\n{c}"
    return f"问题：{q}"


def normalize_prompt(tokenizer, question: str, context: Optional[str], system_prompt: str) -> str:
    user_text = build_user_text(question, context)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_dpo_data(args, tokenizer):
    from datasets import load_dataset

    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file
    raw = load_dataset("json", data_files=data_files)

    def preprocess(example: Dict[str, str]) -> Dict[str, str]:
        prompt = str(example.get(args.prompt_field, "") or "").strip()
        if not prompt:
            q = str(example.get(args.question_field, "") or "").strip()
            c = str(example.get(args.context_field, "") or "").strip()
            prompt = normalize_prompt(tokenizer, q, c, args.system_prompt)

        chosen = str(example.get(args.chosen_field, "") or "").strip()
        rejected = str(example.get(args.rejected_field, "") or "").strip()

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    mapped = raw.map(preprocess, desc="Formatting DPO data")

    def valid_item(x: Dict[str, str]) -> bool:
        return bool((x.get("prompt") or "").strip()) and bool((x.get("chosen") or "").strip()) and bool(
            (x.get("rejected") or "").strip()
        )

    mapped = mapped.filter(valid_item, desc="Filtering invalid rows")

    keep_cols = ["prompt", "chosen", "rejected"]
    train_cols = mapped["train"].column_names
    remove_cols = [c for c in train_cols if c not in keep_cols]
    mapped = mapped.remove_columns(remove_cols)
    return mapped


def load_model_and_tokenizer(args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

    if args.use_lora:
        targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if hasattr(model, "config"):
        model.config.use_cache = False

    return model, tokenizer


def build_training_args(args, has_val: bool):
    from transformers import TrainingArguments
    try:
        from trl import DPOConfig
    except Exception:
        DPOConfig = None

    eval_strategy = "steps" if has_val else "no"

    base_kwargs = {
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
    }

    if eval_strategy == "steps":
        base_kwargs["eval_steps"] = args.eval_steps

    # Prefer DPOConfig on newer TRL, fallback to plain TrainingArguments on older versions.
    arg_cls = DPOConfig if DPOConfig is not None else TrainingArguments

    # Handle versions that renamed evaluation_strategy -> eval_strategy.
    ta_sig = inspect.signature(arg_cls.__init__)
    eval_arg_name = "eval_strategy" if "eval_strategy" in ta_sig.parameters else "evaluation_strategy"
    base_kwargs[eval_arg_name] = eval_strategy

    return arg_cls(**base_kwargs)


def main():
    args = parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from trl import DPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "trl is required for DPO training. Install it first, e.g. `pip install -U trl`."
        ) from exc

    model, tokenizer = load_model_and_tokenizer(args)
    data = load_dpo_data(args, tokenizer)
    train_args = build_training_args(args, has_val=("validation" in data))

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": train_args,
        "train_dataset": data["train"],
        "eval_dataset": data.get("validation"),
    }

    # Keep compatibility with different TRL versions.
    dpo_sig = inspect.signature(DPOTrainer.__init__)

    # Some TRL versions expose these as __init__ kwargs, newer versions move them into args (DPOConfig).
    for k, v in {
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
    }.items():
        if k in dpo_sig.parameters:
            trainer_kwargs[k] = v
        elif hasattr(train_args, k):
            setattr(train_args, k, v)

    # Prefer processing_class (new API). Fallback to tokenizer for older TRL versions.
    try:
        trainer = DPOTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise
        trainer = DPOTrainer(**trainer_kwargs, tokenizer=tokenizer)
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"DPO training done. Model/adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

'''
Example:
python train_qwen25_dpo_lora.py \\
  --model_path qwen2.5-1.7B \\
  --train_file SFT_data/three_high_subset/generated_dpo.jsonl \\
  --output_dir outputs/qwen25_dpo_lora \\
  --use_lora --use_4bit --bf16 \\
  --num_train_epochs 1 \\
  --learning_rate 5e-5 \\
  --per_device_train_batch_size 1 \\
  --gradient_accumulation_steps 16 \\
  --max_length 1024 \\
  --max_prompt_length 512
'''
