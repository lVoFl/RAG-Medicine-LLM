import argparse
import importlib.metadata
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qwen_service.prompts import render_chat_prompt
from qwen_service.rag_runtime import RAGRuntime
from qwen_service.text_utils import clip_text


class ModelRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        if args.cpu:
            self.device = torch.device("cpu")
            print("Detected device: cpu (forced by --cpu)")
        elif torch.cuda.is_available():
            gpu_id = int(args.gpu_id)
            gpu_count = torch.cuda.device_count()
            if gpu_id < 0 or gpu_id >= gpu_count:
                raise ValueError(f"--gpu_id={gpu_id} is out of range. Available GPUs: 0..{gpu_count - 1}")
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Detected device: {self.device} ({torch.cuda.get_device_name(gpu_id)})")
        else:
            self.device = torch.device("cpu")
            print("Detected device: cpu (CUDA not available)")

        print(f"[1/2] Loading tokenizer from: {args.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quantization_config = None
        if args.use_4bit and self.device.type == "cuda":
            try:
                bnb_version = importlib.metadata.version("bitsandbytes")
                self._log(f"bitsandbytes detected: {bnb_version}")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except importlib.metadata.PackageNotFoundError:
                print(
                    "Warning: --use_4bit requested but 'bitsandbytes' is not installed. "
                    "Falling back to non-quantized loading.\n"
                    "Install with: pip install bitsandbytes>=0.43.1"
                )
        elif args.use_4bit and self.device.type != "cuda":
            print("Warning: --use_4bit requires CUDA. Falling back to non-quantized loading on CPU.")

        dtype = torch.float32 if self.device.type == "cpu" else (torch.bfloat16 if args.bf16 else torch.float16)

        print("[2/2] Loading base model")
        device_map = None
        if self.device.type == "cpu":
            device_map = "cpu"
        elif quantization_config is not None:
            # For quantized loading, place the full model on selected GPU explicitly.
            device_map = {"": self.device.index}
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=dtype if quantization_config is None else None,
            device_map=device_map,
        )
        if self.device.type == "cuda" and quantization_config is None:
            base_model = base_model.to(self.device)

        if args.lora_path:
            print(f"Loading LoRA adapter from: {args.lora_path}")
            self.model = PeftModel.from_pretrained(base_model, args.lora_path)
            if self.device.type == "cuda" and quantization_config is None:
                self.model = self.model.to(self.device)
        else:
            print("No LoRA adapter provided. Serving base model only.")
            self.model = base_model
        self.model.eval()
        model_devices = sorted({str(p.device) for p in self.model.parameters()})
        print(f"Model parameter devices: {model_devices}")

        self.rag_runtime: Optional[RAGRuntime] = None
        if args.enable_rag:
            rag_index_dir = Path(args.rag_index_dir)
            self.rag_runtime = RAGRuntime(
                index_dir=rag_index_dir,
                debug_log=args.debug_log,
                debug_max_chars=args.debug_max_chars,
            )
        print("Model runtime is ready.")

    def _log(self, msg: str):
        if self.args.debug_log:
            print(f"[Model] {msg}")

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        history: Optional[list[dict]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        use_rag: Optional[bool] = None,
        rag_top_k: Optional[int] = None,
        rag_candidate_k: Optional[int] = None,
        rag_rerank_top_n: Optional[int] = None,
        rag_rrf_k: Optional[int] = None,
        rag_max_chars_per_doc: Optional[int] = None,
    ):
        start_t = time.time()
        rag_enabled = bool(self.rag_runtime) and (self.args.enable_rag if use_rag is None else bool(use_rag))
        used_context = (context or "").strip()
        retrieved_docs: list[dict] = []
        self._log(
            "generate start | "
            f"question='{clip_text(question, self.args.debug_max_chars)}' "
            f"provided_context_chars={len(used_context)} "
            f"history_turns={len(history or [])} "
            f"use_rag={rag_enabled}"
        )
        if rag_enabled and not used_context:
            used_context, retrieved_docs = self.rag_runtime.retrieve(
                query=question,
                top_k=int(rag_top_k or self.args.rag_top_k),
                candidate_k=int(rag_candidate_k or self.args.rag_candidate_k),
                rerank_top_n=int(rag_rerank_top_n or self.args.rag_rerank_top_n),
                rrf_k=int(rag_rrf_k or self.args.rag_rrf_k),
                max_chars_per_doc=int(rag_max_chars_per_doc or self.args.rag_max_chars_per_doc),
            )
            self._log(
                "rag context built | "
                f"retrieved_docs={len(retrieved_docs)} context_chars={len(used_context)}"
            )
        elif used_context:
            self._log(f"use provided context | context_chars={len(used_context)}")

        prompt = render_chat_prompt(
            tokenizer=self.tokenizer,
            question=question,
            context=used_context,
            history=history,
            system_prompt=system_prompt,
            fallback_system_prompt=self.args.system_prompt,
        )
        gen_max_new_tokens = int(max_new_tokens or self.args.max_new_tokens)
        gen_temperature = float(self.args.temperature if temperature is None else temperature)
        gen_top_p = float(self.args.top_p if top_p is None else top_p)

        with self.lock:
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
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
        self._log(
            "generate done | "
            f"prompt_tokens={int(inputs['input_ids'].shape[-1])} "
            f"completion_tokens={int(new_ids.shape[-1])} "
            f"answer='{clip_text(answer, self.args.debug_max_chars)}' "
            f"elapsed_ms={int((time.time() - start_t) * 1000)}"
        )

        return {
            "answer": answer,
            "context": used_context,
            "retrieved_docs": retrieved_docs,
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(new_ids.shape[-1]),
            },
            "params": {
                "max_new_tokens": gen_max_new_tokens,
                "temperature": gen_temperature,
                "top_p": gen_top_p,
                "use_rag": rag_enabled,
            },
        }
