import json
import os
import select
import subprocess
import tempfile
import threading
import time
from uuid import uuid4
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context

from qwen_service.model_runtime import ModelRuntime
from qwen_service.text_utils import clip_text


def create_app(runtime: ModelRuntime) -> Flask:
    app = Flask(__name__)
    append_lock = threading.Lock()

    def _parse_generate_payload():
        data = request.get_json(silent=True)
        if data is None:
            return None, (jsonify({"error": "invalid json body"}), 400)

        if runtime.args.debug_log:
            safe_data = dict(data)
            for key in ["question", "context", "system_prompt"]:
                if key in safe_data and safe_data[key] is not None:
                    safe_data[key] = clip_text(str(safe_data[key]), runtime.args.debug_max_chars)
            if "history" in safe_data and isinstance(safe_data["history"], list):
                safe_data["history_count"] = len(safe_data["history"])
                safe_data.pop("history", None)
            print(f"[API] request payload: {json.dumps(safe_data, ensure_ascii=False)}")

        question = str(data.get("question", "") or "").strip()
        if not question:
            return None, (jsonify({"error": "question is required"}), 400)

        history = data.get("history")
        if history is not None and not isinstance(history, list):
            return None, (jsonify({"error": "history must be a list"}), 400)

        payload = {
            "question": question,
            "context": data.get("context"),
            "history": history,
            "system_prompt": data.get("system_prompt"),
            "max_new_tokens": data.get("max_new_tokens"),
            "temperature": data.get("temperature"),
            "top_p": data.get("top_p"),
            "use_rag": data.get("use_rag"),
            "rag_top_k": data.get("rag_top_k"),
            "rag_candidate_k": data.get("rag_candidate_k"),
            "rag_rerank_top_n": data.get("rag_rerank_top_n"),
            "rag_rrf_k": data.get("rag_rrf_k"),
            "rag_max_chars_per_doc": data.get("rag_max_chars_per_doc"),
            "rag_min_relevance_score": data.get("rag_min_relevance_score"),
        }
        return payload, None

    def _sse_event(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    @app.get("/health")
    def health_check():
        return jsonify(
            {"status": "ok", "device": str(runtime.device), "base_model": runtime.args.base_model}
        )

    @app.post("/generate")
    def generate_text():
        req_start = time.time()
        payload, error_response = _parse_generate_payload()
        if error_response:
            return error_response

        try:
            result = runtime.generate(**payload)
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

    @app.post("/generate/stream")
    def generate_text_stream():
        req_start = time.time()
        payload, error_response = _parse_generate_payload()
        if error_response:
            return error_response

        @stream_with_context
        def event_stream():
            try:
                for event in runtime.generate_stream(**payload):
                    yield _sse_event(event)
                if runtime.args.debug_log:
                    print(f"[API] /generate/stream done elapsed_ms={int((time.time() - req_start) * 1000)}")
            except Exception as exc:
                yield _sse_event({"type": "error", "error": f"inference failed: {exc}"})

        response = Response(event_stream(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.post("/rag/reload")
    def rag_reload():
        try:
            runtime.reload_rag_index()
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"error": f"rag reload failed: {exc}"}), 500

    @app.post("/faiss/append-text")
    def faiss_append_text():
        req_start = time.time()
        request_id = uuid4().hex[:12]
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "invalid json body"}), 400

        text = str(data.get("text", "") or "").strip()
        source = str(data.get("source", "") or "").strip()
        headings = str(data.get("headings", "") or "")
        category = str(data.get("category", "") or "")
        version = str(data.get("version", "") or "")
        index_dir = str(data.get("index_dir", "") or "").strip()
        max_chars = int(data.get("max_chars", 1000))
        overlap = int(data.get("overlap", 120))
        batch_size = int(data.get("batch_size", 12))
        max_length = int(data.get("max_length", 8192))
        timeout_sec = int(data.get("timeout_sec", 1800))
        debug = bool(data.get("debug", runtime.args.debug_log))

        if not text:
            return jsonify({"error": "text is required"}), 400
        if not source:
            return jsonify({"error": "source is required"}), 400

        project_root = Path(__file__).resolve().parents[2]
        append_script = project_root / "database" / "append_text_to_faiss.py"
        if not append_script.exists():
            return jsonify({"error": f"append script not found: {append_script}"}), 500
        if not index_dir:
            index_dir = str(project_root / "database" / "faiss_index")

        print(
            "[FAISS][%s] received append request | source='%s' text_chars=%d index_dir='%s' "
            "max_chars=%d overlap=%d batch_size=%d max_length=%d timeout_sec=%d"
            % (
                request_id,
                clip_text(source, 120),
                len(text),
                index_dir,
                max_chars,
                overlap,
                batch_size,
                max_length,
                timeout_sec,
            )
        )

        captured_lines: list[str] = []
        with append_lock:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".txt", delete=False
            ) as f:
                f.write(text)
                tmp_path = f.name

            cmd = [
                os.environ.get("PYTHON_BIN", "python"),
                "-u",
                str(append_script),
                "--text-file",
                tmp_path,
                "--source",
                source,
                "--headings",
                headings,
                "--category",
                category,
                "--version",
                version,
                "--index-dir",
                index_dir,
                "--max-chars",
                str(max_chars),
                "--overlap",
                str(overlap),
                "--batch-size",
                str(batch_size),
                "--max-length",
                str(max_length),
            ]
            if debug:
                print(f"[FAISS][{request_id}] subprocess cmd: {cmd}")

            sub_start = time.time()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root / "database"),
                env=env,
            )
            last_output = time.time()
            heartbeat_every = 15

            try:
                while True:
                    if proc.stdout is not None:
                        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                        if ready:
                            line = proc.stdout.readline()
                            if line:
                                clean = line.rstrip("\n")
                                captured_lines.append(clean)
                                print(f"[FAISS][{request_id}] {clean}", flush=True)
                                last_output = time.time()
                                continue

                    ret = proc.poll()
                    if ret is not None:
                        if proc.stdout is not None:
                            rest = proc.stdout.read()
                            if rest:
                                for line in rest.splitlines():
                                    captured_lines.append(line)
                                    print(f"[FAISS][{request_id}] {line}", flush=True)
                        sub_elapsed_ms = int((time.time() - sub_start) * 1000)
                        print(
                            f"[FAISS][{request_id}] subprocess finished rc={ret} "
                            f"elapsed_ms={sub_elapsed_ms}"
                        )
                        break

                    if time.time() - sub_start > timeout_sec:
                        proc.kill()
                        if proc.stdout is not None:
                            try:
                                rest = proc.stdout.read()
                                if rest:
                                    captured_lines.extend(rest.splitlines())
                            except Exception:
                                pass
                        print(f"[FAISS][{request_id}] subprocess timeout after {timeout_sec}s")
                        return (
                            jsonify(
                                {
                                    "request_id": request_id,
                                    "error": "append_text_to_faiss timeout",
                                    "timeout_sec": timeout_sec,
                                    "cmd": cmd,
                                    "stdout": "\n".join(captured_lines)[-2000:],
                                    "hint": (
                                        "首次加载 BAAI/bge-m3 可能较慢，请增大 timeout_sec，"
                                        "或提前在该环境预热/下载模型缓存。"
                                    ),
                                }
                            ),
                            504,
                        )

                    if debug and (time.time() - last_output) >= heartbeat_every:
                        print(
                            f"[FAISS][{request_id}] subprocess still running "
                            f"elapsed_sec={int(time.time() - sub_start)}",
                            flush=True,
                        )
                        last_output = time.time()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        if proc.returncode != 0:
            print(
                f"[FAISS][{request_id}] append failed rc={proc.returncode} "
                f"stderr_tail={(proc.stderr or '')[-500:]}"
            )
            return (
                jsonify(
                    {
                        "request_id": request_id,
                        "error": "append_text_to_faiss failed",
                        "returncode": proc.returncode,
                        "stdout": "\n".join(captured_lines)[-2000:],
                    }
                ),
                500,
            )

        append_result = None
        for line in captured_lines[::-1]:
            line = line.strip()
            if not line:
                continue
            try:
                append_result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if append_result is None:
            append_result = {"ok": True, "stdout": "\n".join(captured_lines).strip()}

        try:
            reload_start = time.time()
            runtime.reload_rag_index()
            reload_elapsed_ms = int((time.time() - reload_start) * 1000)
            rag_reload = {"ok": True}
            print(f"[FAISS][{request_id}] rag reload ok elapsed_ms={reload_elapsed_ms}")
        except Exception as exc:
            rag_reload = {"ok": False, "error": str(exc)}
            print(f"[FAISS][{request_id}] rag reload failed: {exc}")

        total_elapsed_ms = int((time.time() - req_start) * 1000)
        print(f"[FAISS][{request_id}] request done elapsed_ms={total_elapsed_ms}")
        return jsonify(
            {
                "ok": True,
                "request_id": request_id,
                "elapsed_ms": total_elapsed_ms,
                "append_result": append_result,
                "rag_reload": rag_reload,
                "stdout_tail": "\n".join(captured_lines)[-2000:],
            }
        )

    @app.post("/faiss/delete-source")
    def faiss_delete_source():
        req_start = time.time()
        request_id = uuid4().hex[:12]
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "invalid json body"}), 400

        source = str(data.get("source", "") or "").strip()
        if not source:
            return jsonify({"error": "source is required"}), 400

        index_dir = str(data.get("index_dir", "") or "").strip()
        backup = bool(data.get("backup", True))
        timeout_sec = int(data.get("timeout_sec", 1800))

        project_root = Path(__file__).resolve().parents[2]
        script_path = project_root / "database" / "delete_source_from_faiss.py"
        if not script_path.exists():
            return jsonify({"error": f"delete script not found: {script_path}"}), 500
        if not index_dir:
            index_dir = str(project_root / "database" / "faiss_index")

        print(
            f"[FAISS][{request_id}] delete-source request source='{clip_text(source, 120)}' "
            f"index_dir='{index_dir}' backup={backup}"
        )

        cmd = [
            os.environ.get("PYTHON_BIN", "python"),
            "-u",
            str(script_path),
            "--source",
            source,
            "--index-dir",
            index_dir,
        ]
        if backup:
            cmd.append("--backup")

        captured_lines: list[str] = []
        with append_lock:
            sub_start = time.time()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root / "database"),
                env=env,
            )
            while True:
                if proc.stdout is not None:
                    ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                    if ready:
                        line = proc.stdout.readline()
                        if line:
                            clean = line.rstrip("\n")
                            captured_lines.append(clean)
                            print(f"[FAISS][{request_id}] {clean}", flush=True)
                            continue
                ret = proc.poll()
                if ret is not None:
                    if proc.stdout is not None:
                        rest = proc.stdout.read()
                        if rest:
                            for line in rest.splitlines():
                                captured_lines.append(line)
                                print(f"[FAISS][{request_id}] {line}", flush=True)
                    print(
                        f"[FAISS][{request_id}] delete subprocess finished rc={ret} "
                        f"elapsed_ms={int((time.time() - sub_start) * 1000)}"
                    )
                    break
                if time.time() - sub_start > timeout_sec:
                    proc.kill()
                    return (
                        jsonify(
                            {
                                "request_id": request_id,
                                "error": "delete_source_from_faiss timeout",
                                "timeout_sec": timeout_sec,
                                "cmd": cmd,
                                "stdout": "\n".join(captured_lines)[-2000:],
                            }
                        ),
                        504,
                    )

        if proc.returncode != 0:
            return (
                jsonify(
                    {
                        "request_id": request_id,
                        "error": "delete_source_from_faiss failed",
                        "returncode": proc.returncode,
                        "stdout": "\n".join(captured_lines)[-2000:],
                    }
                ),
                500,
            )

        delete_result = None
        for line in captured_lines[::-1]:
            line = line.strip()
            if not line:
                continue
            try:
                delete_result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if delete_result is None:
            delete_result = {"ok": True, "stdout": "\n".join(captured_lines).strip()}

        try:
            runtime.reload_rag_index()
            rag_reload = {"ok": True}
        except Exception as exc:
            rag_reload = {"ok": False, "error": str(exc)}

        return jsonify(
            {
                "ok": True,
                "request_id": request_id,
                "elapsed_ms": int((time.time() - req_start) * 1000),
                "delete_result": delete_result,
                "rag_reload": rag_reload,
            }
        )

    return app
