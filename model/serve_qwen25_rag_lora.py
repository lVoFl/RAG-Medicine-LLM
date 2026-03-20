"""
部署脚本：使用 Flask 启动本地推理服务，供 Node.js server 调用。

启动示例：
python serve_qwen25_rag_lora.py --base_model model --lora_path outputs/qwen25_rag_lora --bf16 --port 8001
"""

import argparse
import threading
from typing import Optional

import torch
from flask import Flask, jsonify, request
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_SYSTEM_PROMPT = "你是一个严谨的医疗助手，请基于提供的检索资料回答问题。若资料不足，明确说明不确定。"


class ModelRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()

        print(f"[1/2] Loading tokenizer from: {args.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quantization_config = None
        if args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        dtype = torch.bfloat16 if args.bf16 else torch.float16

        print(f"[2/2] Loading base model and LoRA adapter from: {args.lora_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=dtype if not args.use_4bit else None,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, args.lora_path)
        self.model.eval()
        print("Model runtime is ready.")

    def build_prompt(self, question: str, context: Optional[str], system_prompt: Optional[str]) -> str:
        question = (question or "").strip()
        context = (context or "").strip()
        effective_system_prompt = (system_prompt or "").strip() or self.args.system_prompt
        user_text = f"问题：{question}\n\n检索资料：\n{context}" if context else f"问题：{question}"
        messages = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        prompt = self.build_prompt(question=question, context=context, system_prompt=system_prompt)
        gen_max_new_tokens = int(max_new_tokens or self.args.max_new_tokens)
        gen_temperature = float(self.args.temperature if temperature is None else temperature)
        gen_top_p = float(self.args.top_p if top_p is None else top_p)

        with self.lock:
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_max_new_tokens,
                temperature=gen_temperature,
                top_p=gen_top_p,
                do_sample=gen_temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(new_ids.shape[-1]),
            },
            "params": {
                "max_new_tokens": gen_max_new_tokens,
                "temperature": gen_temperature,
                "top_p": gen_top_p,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Qwen2.5 RAG-LoRA model via Flask")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--base_model", type=str, default="model", help="Base model path")
    parser.add_argument("--lora_path", type=str, default="outputs/qwen25_rag_lora", help="LoRA adapter path")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def create_app(runtime: ModelRuntime) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/generate")
    def generate_text():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "invalid json body"}), 400

        question = str(data.get("question", "") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400

        try:
            result = runtime.generate(
                question=question,
                context=data.get("context"),
                system_prompt=data.get("system_prompt"),
                max_new_tokens=data.get("max_new_tokens"),
                temperature=data.get("temperature"),
                top_p=data.get("top_p"),
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": f"inference failed: {exc}"}), 500

    return app


def main():
    args = parse_args()
    runtime = ModelRuntime(args)
    app = create_app(runtime)

    print(f"Model service listening on http://{args.host}:{args.port}")
    print("Health endpoint: GET /health")
    print("Generate endpoint: POST /generate")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
