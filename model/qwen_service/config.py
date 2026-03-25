import argparse
from pathlib import Path

from qwen_service.prompts import DEFAULT_SYSTEM_PROMPT

PKG_DIR = Path(__file__).resolve().parent
MODEL_DIR = PKG_DIR.parent
PROJECT_ROOT = MODEL_DIR.parent

DEFAULT_BASE_MODEL = str(MODEL_DIR / "qwen2.5-1.7B")
DEFAULT_LORA_PATH = str(MODEL_DIR / "outputs" / "qwen25_rag_lora")
DEFAULT_RAG_INDEX_DIR = str(PROJECT_ROOT / "database" / "faiss_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Qwen2.5 RAG-LoRA model via Flask")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="Base model path")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter path, optional")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA GPU index when not using --cpu")
    parser.add_argument("--enable_rag", action="store_true", help="Enable built-in RAG retrieval")
    parser.add_argument("--rag_index_dir", type=str, default=DEFAULT_RAG_INDEX_DIR)
    parser.add_argument("--rag_top_k", type=int, default=5)
    parser.add_argument("--rag_candidate_k", type=int, default=50)
    parser.add_argument("--rag_rerank_top_n", type=int, default=20)
    parser.add_argument("--rag_rrf_k", type=int, default=60)
    parser.add_argument("--rag_max_chars_per_doc", type=int, default=500)
    parser.add_argument("--debug_log", action="store_true", help="Print debug logs for request/rag/generation")
    parser.add_argument("--debug_max_chars", type=int, default=800, help="Max chars to print in debug logs")
    parser.add_argument(
        "--with_lora",
        action="store_true",
        help=f"Enable default LoRA adapter: {DEFAULT_LORA_PATH}",
    )

    args = parser.parse_args()
    if args.with_lora and args.lora_path is None:
        args.lora_path = DEFAULT_LORA_PATH
    return args
