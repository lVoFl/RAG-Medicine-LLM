from typing import Optional


DEFAULT_SYSTEM_PROMPT = (
    "你是一名严谨、审慎的医疗问答助手。\n\n"
    "请优先基于“检索资料”回答问题，并结合必要的医学常识进行补充说明，以确保回答完整、准确。\n\n"
    "要求：\n"
    "1. 不得编造与资料或医学共识相矛盾的内容。\n\n"
    "2. 请判断检索资料与问题的相关性：\n"
    "- 若高度相关：以资料为主回答。\n"
    "- 若部分相关：结合资料与医学常识补充。\n"
    "- 若相关性较低：以通用医学知识为主，并说明“检索资料未充分覆盖该问题”。\n\n"
    "3. 若问题较宽泛（如疾病名称）：\n"
    "- 先进行简要概述（定义/常见表现）。\n"
    "- 再结合检索资料重点展开。\n\n"
    "4. 回答使用markdown格式，内容应清晰、具体、通俗易懂。\n\n"
    "5. 涉及治疗或用药时：\n"
    "- 避免给出绝对化或个体化用药方案。\n"
    "- 若缺少年龄、体重、肝肾功能、合并用药、过敏史等关键信息，应先提示补充。\n\n"
    "6. 若出现急危重症信号（如胸痛、呼吸困难、意识改变、持续高热、明显出血等），应优先建议立即线下就医或急诊。\n\n"
    "7. 当检索资料不足以完整回答问题时：\n"
    "- 可以基于通用医学知识进行补充。\n"
    "- 并适当说明信息来源类型（如“通用医学知识”）。"
)


def build_user_text(question: str, context: Optional[str]) -> str:
    clean_question = (question or "").strip()
    clean_context = (context or "").strip()
    if clean_context:
        return f"问题：{clean_question}\n\n检索资料：\n{clean_context}"
    return f"问题：{clean_question}"


def build_chat_messages(
    question: str,
    context: Optional[str],
    system_prompt: Optional[str],
    fallback_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    effective_system_prompt = (system_prompt or "").strip() or fallback_system_prompt
    return [
        {"role": "system", "content": effective_system_prompt},
        {"role": "user", "content": build_user_text(question=question, context=context)},
    ]


def render_chat_prompt(
    tokenizer,
    question: str,
    context: Optional[str],
    system_prompt: Optional[str],
    fallback_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    messages = build_chat_messages(
        question=question,
        context=context,
        system_prompt=system_prompt,
        fallback_system_prompt=fallback_system_prompt,
    )
    print("messages: ")
    print(messages)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
