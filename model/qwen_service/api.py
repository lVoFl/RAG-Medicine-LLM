import json
import time

from flask import Flask, jsonify, request

from qwen_service.model_runtime import ModelRuntime
from qwen_service.text_utils import clip_text


def create_app(runtime: ModelRuntime) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health_check():
        return jsonify(
            {"status": "ok", "device": str(runtime.device), "base_model": runtime.args.base_model}
        )

    @app.post("/generate")
    def generate_text():
        req_start = time.time()
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "invalid json body"}), 400
        if runtime.args.debug_log:
            safe_data = dict(data)
            for key in ["question", "context", "system_prompt"]:
                if key in safe_data and safe_data[key] is not None:
                    safe_data[key] = clip_text(str(safe_data[key]), runtime.args.debug_max_chars)
            if "history" in safe_data and isinstance(safe_data["history"], list):
                safe_data["history_count"] = len(safe_data["history"])
                safe_data.pop("history", None)
            print(f"[API] /generate request: {json.dumps(safe_data, ensure_ascii=False)}")

        question = str(data.get("question", "") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400

        try:
            history = data.get("history")
            if history is not None and not isinstance(history, list):
                return jsonify({"error": "history must be a list"}), 400

            result = runtime.generate(
                question=question,
                context=data.get("context"),
                history=history,
                system_prompt=data.get("system_prompt"),
                max_new_tokens=data.get("max_new_tokens"),
                temperature=data.get("temperature"),
                top_p=data.get("top_p"),
                use_rag=data.get("use_rag"),
                rag_top_k=data.get("rag_top_k"),
                rag_candidate_k=data.get("rag_candidate_k"),
                rag_rerank_top_n=data.get("rag_rerank_top_n"),
                rag_rrf_k=data.get("rag_rrf_k"),
                rag_max_chars_per_doc=data.get("rag_max_chars_per_doc"),
            )
            if runtime.args.debug_log:
                print(
                    "[API] /generate response: "
                    f"answer='{clip_text(result.get('answer', ''), runtime.args.debug_max_chars)}' "
                    f"retrieved_docs={len(result.get('retrieved_docs', []))} "
                    f"elapsed_ms={int((time.time() - req_start) * 1000)}"
                )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": f"inference failed: {exc}"}), 500

    return app

