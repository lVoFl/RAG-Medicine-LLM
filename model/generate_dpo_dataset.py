import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib import error, request

from qwen_service.prompts import DEFAULT_SYSTEM_PROMPT


CHOSEN_SYSTEM_PROMPT = (
    "你是资深医疗问答助手。你需要基于用户问题和检索资料，生成高质量、审慎、安全的回答。"
    "回答必须专业、清晰、可执行，避免绝对化与不确定内容的臆断。"
)

CHOSEN_USER_TEMPLATE = """请根据以下输入生成“高质量回答（chosen）”。

要求：
1. 优先使用检索资料中的信息，必要时用通用医学知识补充。
2. 如果信息不足，明确指出不确定性与需补充的信息。
3. 涉及治疗/用药时避免给出过于个体化且高风险的具体处方。
4. 遇到急危重症信号需提示及时线下就医。
5. 使用中文，结构化表达（可用小标题或分点）。

输入：
{input_block}

仅输出最终回答正文，不要输出解释。"""

REJECTED_SYSTEM_PROMPT = (
    "你要生成一个‘看起来像正常回答但质量较差’的医疗回答，用于偏好学习反例。"
    "不要写得荒谬，保持语言流畅，但应体现常见缺陷。"
)

REJECTED_USER_TEMPLATE = """请根据以下输入生成“较差回答（rejected）”。

缺陷模式（至少满足两项）：
1. 对不确定结论过度自信，不提示边界条件。
2. 过度泛化，忽略检索资料中的关键细节。
3. 安全意识不足，不提醒风险或就医时机。
4. 回答笼统，缺少针对性。

限制：
- 不要加入明显违法或极端危险内容。
- 不要声明自己在故意生成差答案。
- 使用中文，长度与正常回答接近。

输入：
{input_block}

仅输出最终回答正文，不要输出解释。"""

JUDGE_SYSTEM_PROMPT = (
    "你是严格的医疗问答评审员。比较两个回答质量，返回 JSON。"
    "评估维度：事实一致性、安全性、相关性、完整性、清晰度。"
)

JUDGE_USER_TEMPLATE = """请比较以下两个回答，判断哪个更好。

题目：
{prompt}

回答A：
{answer_a}

回答B：
{answer_b}

输出严格 JSON，格式：
{{
  "winner": "A" 或 "B" 或 "TIE",
  "score_a": 1-10 的整数,
  "score_b": 1-10 的整数,
  "reason": "一句话理由"
}}

仅输出 JSON。"""


@dataclass
class ChatClient:
    api_base: str
    api_key: Optional[str]
    timeout: int = 120
    proxy: Optional[str] = None
    use_env_proxy: bool = False

    def _open(self, req: request.Request):
        if self.proxy:
            opener = request.build_opener(
                request.ProxyHandler({"http": self.proxy, "https": self.proxy})
            )
            return opener.open(req, timeout=self.timeout)
        if self.use_env_proxy:
            return request.urlopen(req, timeout=self.timeout)
        opener = request.build_opener(request.ProxyHandler({}))
        return opener.open(req, timeout=self.timeout)

    def _resolve_url(self, path: str) -> str:
        base = self.api_base
        clean_path = "/" + path.lstrip("/")
        # Respect user-provided endpoint as-is when it already points to an API route.
        if "/chat/completions" in base or "/responses" in base:
            return base
        return base.rstrip("/") + clean_path

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._resolve_url(path)
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with self._open(req) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} calling {url}: {body[:500]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Network error calling {url}: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            # Some gateways return SSE payloads (event:/data:) even when non-streaming is expected.
            sse_obj = self._try_parse_sse_json(body)
            if sse_obj is not None:
                return sse_obj
            raise RuntimeError(f"Invalid JSON from {url}: {body[:500]}") from exc

    def _get_json_by_url(self, url: str) -> Dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(url=url, headers=headers, method="GET")
        try:
            with self._open(req) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} calling {url}: {body[:500]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Network error calling {url}: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            sse_obj = self._try_parse_sse_json(body)
            if sse_obj is not None:
                return sse_obj
            raise RuntimeError(f"Invalid JSON from {url}: {body[:500]}") from exc

    @staticmethod
    def _try_parse_sse_json(body: str) -> Optional[Dict[str, Any]]:
        if "data:" not in body:
            return None
        objs: list[Dict[str, Any]] = []
        text_chunks: list[str] = []
        last_response: Optional[Dict[str, Any]] = None
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                objs.append(obj)
                ev_type = str(obj.get("type", "")).strip()
                # Common Responses API streaming events.
                if ev_type == "response.output_text.delta":
                    delta = obj.get("delta")
                    if isinstance(delta, str) and delta:
                        text_chunks.append(delta)
                elif ev_type == "response.output_text.done":
                    text = obj.get("text")
                    if isinstance(text, str) and text:
                        text_chunks.append(text)
                elif ev_type == "response.content_part.added":
                    part = obj.get("part")
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str) and t:
                            text_chunks.append(t)
                elif ev_type == "response.output_item.added":
                    item = obj.get("item")
                    if isinstance(item, dict):
                        for c in item.get("content", []) or []:
                            if not isinstance(c, dict):
                                continue
                            t = c.get("text")
                            if isinstance(t, str) and t:
                                text_chunks.append(t)

                resp = obj.get("response")
                if isinstance(resp, dict):
                    last_response = resp
        if not objs:
            return None
        merged_text = "".join(text_chunks).strip()
        if merged_text:
            out: Dict[str, Any] = {"output_text": merged_text}
            if last_response is not None:
                out["response"] = last_response
            return out
        # Prefer final completed frame, otherwise use last frame.
        for obj in reversed(objs):
            resp = obj.get("response")
            if isinstance(resp, dict) and resp.get("status") == "completed":
                return {"response": resp}
        return objs[-1]

    def _is_responses_api(self) -> bool:
        base = (self.api_base or "").lower()
        return "/responses" in base

    @staticmethod
    def _messages_to_responses_input(messages: list[dict]) -> list[dict]:
        system_instructions: list[str] = []
        non_system_msgs: list[dict] = []

        for m in messages or []:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user")).strip() or "user"
            content = str(m.get("content", "") or "").strip()
            if not content:
                continue
            if role == "system":
                system_instructions.append(content)
            else:
                non_system_msgs.append({"role": role, "content": content})

        # Some gateways reject system role in Responses API.
        # Merge system guidance into the first user message to preserve behavior.
        if system_instructions:
            merged_sys = "\n\n".join(system_instructions)
            prefix = f"请严格遵循以下系统指令：\n{merged_sys}\n\n"
            if non_system_msgs:
                non_system_msgs[0]["content"] = prefix + non_system_msgs[0]["content"]
            else:
                non_system_msgs.append({"role": "user", "content": prefix})

        out: list[dict] = []
        for m in non_system_msgs:
            role = str(m.get("role", "user")).strip() or "user"
            content = str(m.get("content", "") or "").strip()
            out.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text",
                            "text": content,
                        }
                    ],
                }
            )
        return out

    @staticmethod
    def _extract_text_from_responses(out: Dict[str, Any]) -> str:
        # Prefer top-level shortcut when provided by gateway.
        top = out.get("output_text")
        if isinstance(top, str) and top.strip():
            return top.strip()
        if isinstance(top, list):
            merged = "\n".join(str(x) for x in top if str(x).strip()).strip()
            if merged:
                return merged

        # Then try nested response payload.
        if isinstance(out.get("response"), dict):
            out = out["response"]
            top2 = out.get("output_text")
            if isinstance(top2, str) and top2.strip():
                return top2.strip()
            if isinstance(top2, list):
                merged2 = "\n".join(str(x) for x in top2 if str(x).strip()).strip()
                if merged2:
                    return merged2

        texts: list[str] = []
        for item in out.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for c in item.get("content", []) or []:
                if not isinstance(c, dict):
                    continue
                # Common fields in Responses-style outputs.
                if c.get("type") in {"output_text", "text", "input_text"}:
                    t = c.get("text")
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
                elif isinstance(c.get("text"), str) and c.get("text", "").strip():
                    texts.append(c["text"].strip())
        return "\n".join(texts).strip()

    def _resolve_response_retrieve_url(self, response_id: str) -> Optional[str]:
        base = (self.api_base or "").rstrip("/")
        if not response_id:
            return None
        if "/responses" in base:
            # If base is already .../responses, append /{id}
            if base.endswith("/responses"):
                return f"{base}/{response_id}"
            # If base contains extra segments after /responses, fallback to standard root.
            prefix = base.split("/responses", 1)[0]
            return f"{prefix}/responses/{response_id}"
        if base.endswith("/v1"):
            return f"{base}/responses/{response_id}"
        return f"{base}/responses/{response_id}"

    def _wait_response_completed(self, out: Dict[str, Any]) -> Dict[str, Any]:
        response = out.get("response") if isinstance(out.get("response"), dict) else out
        if not isinstance(response, dict):
            return out
        status = str(response.get("status", "")).lower()
        if status in {"completed", "failed", "cancelled"}:
            return out
        response_id = str(response.get("id", "")).strip()
        retrieve_url = self._resolve_response_retrieve_url(response_id)
        if not retrieve_url:
            return out

        deadline = time.time() + max(5, self.timeout)
        while time.time() < deadline:
            polled = self._get_json_by_url(retrieve_url)
            polled_response = (
                polled.get("response") if isinstance(polled.get("response"), dict) else polled
            )
            if not isinstance(polled_response, dict):
                time.sleep(0.8)
                continue
            st = str(polled_response.get("status", "")).lower()
            if st in {"completed", "failed", "cancelled"}:
                return polled
            time.sleep(0.8)
        return out

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        retries: int = 2,
    ) -> str:
        use_responses_api = self._is_responses_api()
        if use_responses_api:
            payload = {
                "model": model,
                "input": self._messages_to_responses_input(messages),
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "stream": False,
            }
            path = "/responses"
        else:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            path = "/chat/completions"
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                out = self._post_json(path, payload)
                if use_responses_api:
                    out = self._wait_response_completed(out)
                    content = self._extract_text_from_responses(out)
                    if not content:
                        raise RuntimeError(f"No output text returned: {out}")
                    return content

                choices = out.get("choices") or []
                if not choices:
                    raise RuntimeError(f"No choices returned: {out}")
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            texts.append(item.get("text", ""))
                    content = "\n".join(texts)
                return str(content).strip()
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
        raise RuntimeError(f"Chat request failed after {retries + 1} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-quality DPO dataset with LLM + judge filtering")
    parser.add_argument(
        "--input_file",
        type=str,
        default="SFT_data/three_high_subset/merged_three_high_qa.dedup.jsonl",
        help="Input jsonl with question/context/answer",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="SFT_data/three_high_subset/generated_dpo.jsonl",
        help="Output jsonl with prompt/chosen/rejected",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key; fallback to DASHSCOPE_API_KEY, then OPENAI_API_KEY",
    )
    parser.add_argument("--gen_model", type=str, required=True, help="Model for chosen/rejected generation")
    parser.add_argument("--judge_model", type=str, default=None, help="Model for pairwise judging")
    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--context_field", type=str, default="context")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in input file")
    parser.add_argument("--resume", action="store_true", help="Skip prompts already present in output file")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between samples")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Optional HTTP(S) proxy URL, e.g. http://127.0.0.1:7890",
    )
    parser.add_argument(
        "--use_env_proxy",
        action="store_true",
        help="Use system/environment proxy settings (disabled by default to avoid Windows proxy hang)",
    )
    parser.add_argument("--gen_max_tokens", type=int, default=700)
    parser.add_argument("--judge_max_tokens", type=int, default=220)
    parser.add_argument("--retries", type=int, default=1, help="Retry count per request")
    parser.add_argument("--min_chosen_score", type=int, default=7)
    parser.add_argument("--min_score_gap", type=int, default=2)
    parser.add_argument("--max_similarity", type=float, default=0.92)
    parser.add_argument("--max_question_chars", type=int, default=500)
    parser.add_argument("--max_context_chars", type=int, default=2400)
    parser.add_argument("--max_reference_chars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="Print runtime profiling summary")
    parser.add_argument("--profile_top_n", type=int, default=5, help="Top N slow samples in profiling")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text


def clip_text(text: str, limit: int) -> str:
    s = (text or "").strip()
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + "\n...[truncated]"


def build_prompt(question: str, context: str, max_question_chars: int, max_context_chars: int) -> str:
    q = clip_text(question, max_question_chars)
    c = clip_text(context, max_context_chars)
    if c:
        return f"问题：{q}\n\n检索资料：\n{c}"
    return f"问题：{q}"


def build_input_block(
    question: str,
    context: str,
    reference_answer: str,
    max_question_chars: int,
    max_context_chars: int,
    max_reference_chars: int,
) -> str:
    q = clip_text(question, max_question_chars)
    c = clip_text(context, max_context_chars)
    r = clip_text(reference_answer, max_reference_chars)

    parts = [f"问题：{q}"]
    if c:
        parts.append(f"检索资料：\n{c}")
    if r:
        parts.append(f"参考答案（可参考但不要照抄）：\n{r}")
    return "\n\n".join(parts)


def parse_judge_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()


def load_existing_prompts(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    for item in read_jsonl(output_path):
        prompt = str(item.get("prompt", "")).strip()
        if prompt:
            done.add(prompt)
    return done


def summarize_times(values: list[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "sum": 0.0}
    arr = sorted(values)
    n = len(arr)

    def pct(p: float) -> float:
        if n == 1:
            return arr[0]
        idx = int(round((n - 1) * p))
        idx = max(0, min(n - 1, idx))
        return arr[idx]

    return {
        "count": n,
        "avg": sum(arr) / n,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": arr[-1],
        "sum": sum(arr),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_key = (raw_api_key or "").strip().strip("'").strip('"')
    if api_key.lower().startswith("bearer "):
        api_key = api_key[7:].strip()
    if not api_key:
        raise ValueError(
            "API key is empty. Please set --api_key or environment variable DASHSCOPE_API_KEY."
        )
    judge_model = args.judge_model or args.gen_model
    client = ChatClient(
        api_base=args.api_base,
        api_key=api_key,
        timeout=args.timeout,
        proxy=args.proxy,
        use_env_proxy=args.use_env_proxy,
    )

    load_t0 = time.perf_counter()
    rows = list(read_jsonl(input_path))
    load_sec = time.perf_counter() - load_t0
    if args.start_index > 0:
        rows = rows[args.start_index :]
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    resume_t0 = time.perf_counter()
    done_prompts = load_existing_prompts(output_path) if args.resume else set()
    resume_sec = time.perf_counter() - resume_t0

    kept = 0
    seen = 0
    skipped = 0
    skip_reasons: dict[str, int] = defaultdict(int)
    profile_times: dict[str, list[float]] = defaultdict(list)
    sample_profiles: list[dict[str, float]] = []
    run_t0 = time.perf_counter()

    with output_path.open("a", encoding="utf-8") as out_f:
        for item in rows:
            sample_t0 = time.perf_counter()
            seen += 1
            question = str(item.get(args.question_field, "") or "").strip()
            context = str(item.get(args.context_field, "") or "").strip()
            ref_answer = str(item.get(args.answer_field, "") or "").strip()
            if not question:
                skipped += 1
                skip_reasons["empty_question"] += 1
                profile_times["sample_total"].append(time.perf_counter() - sample_t0)
                continue

            prompt = build_prompt(
                question=question,
                context=context,
                max_question_chars=args.max_question_chars,
                max_context_chars=args.max_context_chars,
            )
            if prompt in done_prompts:
                skipped += 1
                skip_reasons["resume_duplicate_prompt"] += 1
                profile_times["sample_total"].append(time.perf_counter() - sample_t0)
                continue

            input_block = build_input_block(
                question=question,
                context=context,
                reference_answer=ref_answer,
                max_question_chars=args.max_question_chars,
                max_context_chars=args.max_context_chars,
                max_reference_chars=args.max_reference_chars,
            )
            print(f"[sample {seen}] start", flush=True)

            try:
                t0 = time.time()
                print(
                    f"[sample {seen}] chosen request "
                    f"(prompt_chars={len(input_block)}, max_tokens={args.gen_max_tokens})",
                    flush=True,
                )
                chosen = client.chat(
                    model=args.gen_model,
                    messages=[
                        {"role": "system", "content": CHOSEN_SYSTEM_PROMPT + "\n" + DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": CHOSEN_USER_TEMPLATE.format(input_block=input_block)},
                    ],
                    temperature=0.3,
                    max_tokens=args.gen_max_tokens,
                    retries=args.retries,
                )
                chosen_sec = time.time() - t0
                profile_times["chosen"].append(chosen_sec)
                print(f"[sample {seen}] chosen done ({chosen_sec:.1f}s)", flush=True)

                t1 = time.time()
                print(
                    f"[sample {seen}] rejected request "
                    f"(prompt_chars={len(input_block)}, max_tokens={args.gen_max_tokens})",
                    flush=True,
                )
                rejected = client.chat(
                    model=args.gen_model,
                    messages=[
                        {"role": "system", "content": REJECTED_SYSTEM_PROMPT},
                        {"role": "user", "content": REJECTED_USER_TEMPLATE.format(input_block=input_block)},
                    ],
                    temperature=0.8,
                    max_tokens=args.gen_max_tokens,
                    retries=args.retries,
                )
                rejected_sec = time.time() - t1
                profile_times["rejected"].append(rejected_sec)
                print(f"[sample {seen}] rejected done ({rejected_sec:.1f}s)", flush=True)

                if not chosen or not rejected:
                    skipped += 1
                    skip_reasons["empty_chosen_or_rejected"] += 1
                    profile_times["sample_total"].append(time.perf_counter() - sample_t0)
                    continue

                sim = text_similarity(chosen, rejected)
                if sim >= args.max_similarity:
                    skipped += 1
                    skip_reasons["similarity_too_high"] += 1
                    profile_times["sample_total"].append(time.perf_counter() - sample_t0)
                    continue

                t2 = time.time()
                print(
                    f"[sample {seen}] judge request (prompt_chars={len(prompt)}, max_tokens={args.judge_max_tokens})",
                    flush=True,
                )
                judge_raw = client.chat(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": JUDGE_USER_TEMPLATE.format(
                                prompt=prompt,
                                answer_a=chosen,
                                answer_b=rejected,
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=args.judge_max_tokens,
                    retries=args.retries,
                )
                judge_sec = time.time() - t2
                profile_times["judge"].append(judge_sec)
                print(f"[sample {seen}] judge done ({judge_sec:.1f}s)", flush=True)

                judge = parse_judge_json(judge_raw)
                if not judge:
                    skipped += 1
                    skip_reasons["judge_json_parse_failed"] += 1
                    profile_times["sample_total"].append(time.perf_counter() - sample_t0)
                    continue

                winner = str(judge.get("winner", "")).upper().strip()
                score_a = int(judge.get("score_a", 0))
                score_b = int(judge.get("score_b", 0))
                if winner != "A":
                    skipped += 1
                    skip_reasons["judge_winner_not_A"] += 1
                    profile_times["sample_total"].append(time.perf_counter() - sample_t0)
                    continue
                if score_a < args.min_chosen_score:
                    skipped += 1
                    skip_reasons["chosen_score_too_low"] += 1
                    profile_times["sample_total"].append(time.perf_counter() - sample_t0)
                    continue
                if score_a - score_b < args.min_score_gap:
                    skipped += 1
                    skip_reasons["score_gap_too_small"] += 1
                    profile_times["sample_total"].append(time.perf_counter() - sample_t0)
                    continue

                out = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "question": question,
                        "context": context,
                        "source_answer": ref_answer,
                        "judge": {
                            "winner": winner,
                            "score_a": score_a,
                            "score_b": score_b,
                            "reason": str(judge.get("reason", "")),
                        },
                        "similarity": round(sim, 4),
                    },
                }
                t_write = time.perf_counter()
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                out_f.flush()
                write_sec = time.perf_counter() - t_write
                profile_times["write_flush"].append(write_sec)
                kept += 1
                done_prompts.add(prompt)
                sample_total_sec = time.perf_counter() - sample_t0
                profile_times["sample_total"].append(sample_total_sec)
                sample_profiles.append(
                    {
                        "sample": float(seen),
                        "total": sample_total_sec,
                        "chosen": chosen_sec,
                        "rejected": rejected_sec,
                        "judge": judge_sec,
                        "write_flush": write_sec,
                    }
                )

                if seen % 10 == 0:
                    print(f"[progress] seen={seen} kept={kept} skipped={skipped}")
            except Exception as exc:
                skipped += 1
                skip_reasons["exception"] += 1
                profile_times["sample_total"].append(time.perf_counter() - sample_t0)
                print(f"[warn] sample {seen} failed: {exc}")

            if args.sleep > 0:
                sleep_t0 = time.perf_counter()
                time.sleep(args.sleep)
                profile_times["sleep"].append(time.perf_counter() - sleep_t0)

    run_total_sec = time.perf_counter() - run_t0
    print(f"Done. seen={seen}, kept={kept}, skipped={skipped}, output={output_path}")
    if skip_reasons:
        print("Skip reasons:")
        for reason, cnt in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {reason}: {cnt}")
    if args.profile:
        print("\n=== Profiling Summary ===")
        print(
            f"startup: load_jsonl={load_sec:.3f}s, load_resume={resume_sec:.3f}s, total_run={run_total_sec:.3f}s"
        )

        stage_names = ["chosen", "rejected", "judge", "write_flush", "sleep", "sample_total"]
        stage_sums = {}
        for stage in stage_names:
            stats = summarize_times(profile_times.get(stage, []))
            stage_sums[stage] = stats["sum"]
        denom = sum(stage_sums.values()) or 1.0

        for stage in stage_names:
            stats = summarize_times(profile_times.get(stage, []))
            if stats["count"] == 0:
                continue
            ratio = (stage_sums[stage] / denom) * 100.0
            print(
                f"{stage:12s} count={int(stats['count']):4d} "
                f"avg={stats['avg']:.3f}s p50={stats['p50']:.3f}s "
                f"p90={stats['p90']:.3f}s max={stats['max']:.3f}s "
                f"sum={stats['sum']:.3f}s ({ratio:.1f}%)"
            )

        top_n = max(1, args.profile_top_n)
        if sample_profiles:
            print(f"\nTop {top_n} slow samples:")
            slow = sorted(sample_profiles, key=lambda x: x["total"], reverse=True)[:top_n]
            for item in slow:
                print(
                    f"sample={int(item['sample'])} total={item['total']:.3f}s "
                    f"chosen={item['chosen']:.3f}s rejected={item['rejected']:.3f}s "
                    f"judge={item['judge']:.3f}s write_flush={item['write_flush']:.4f}s"
                )


if __name__ == "__main__":
    main()
