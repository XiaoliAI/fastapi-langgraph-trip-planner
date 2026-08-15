import json
import logging
import re
from typing import Any, Optional

from ..models.schemas import TripChangeIntent
from .llm_service import get_chat_model


logger = logging.getLogger(__name__)


SMALL_CHANGE_KEYWORDS = [
    "删除",
    "去掉",
    "删掉",
    "换掉",
    "替换",
    "改成",
    "换成",
    "不要",
    "餐",
    "早餐",
    "午餐",
    "晚餐",
    "酒店",
    "住宿",
]

MAJOR_REVISION_KEYWORDS = [
    "重新规划",
    "重做",
    "整体",
    "少走路",
    "轻松",
    "不要太累",
    "太赶",
    "舒服",
    "亲子",
    "老人",
    "预算控制",
    "便宜点",
    "高端点",
    "主题",
]


def classify_with_rules(message: str) -> TripChangeIntent:
    text = message.strip()

    if not text:
        return TripChangeIntent(
            change_type="clarification_needed",
            summary="用户没有提供明确修改内容",
            clarification_question="你想调整行程的哪一部分？例如景点、餐饮、酒店、预算或整体节奏。",
        )

    if any(keyword in text for keyword in MAJOR_REVISION_KEYWORDS):
        return TripChangeIntent(
            change_type="major_revision",
            summary=f"用户希望对行程进行整体调整：{text}",
        )

    if any(keyword in text for keyword in SMALL_CHANGE_KEYWORDS):
        return TripChangeIntent(
            change_type="small_change",
            summary=f"用户希望进行局部修改：{text}",
            patch_operations=[],
        )

    return TripChangeIntent(
        change_type="clarification_needed",
        summary=f"用户表达不够具体：{text}",
        clarification_question="你希望我具体调整哪一部分？例如删除某个景点、替换餐厅、调整酒店，还是重新规划整体路线？",
    )


def build_intent_classification_prompt(message: str, knowledge_text: str = "") -> str:
    return f"""
你正在判断用户对现有旅行计划的修改意图。

只返回一个 JSON 对象，不要返回 Markdown，不要返回解释文字。

JSON 字段要求：
- change_type: 只能是 "small_change", "major_revision", "clarification_needed" 之一
- summary: 必须使用中文，简短概括用户的修改意图
- patch_operations: 数组，不确定时返回 []
- clarification_question: 需要继续追问时返回中文问题，否则返回 null

分类规则：
- 对删除、替换某个具体景点、餐厅、酒店等局部修改，使用 "small_change"。
- 对降低步行强度、亲子友好、预算变化、节奏变化、主题变化等整体行程调整，使用 "major_revision"。
- 如果用户表达过于模糊，无法确定要修改什么，使用 "clarification_needed"。

输出语言要求：
- summary 必须使用中文。
- clarification_question 必须使用中文。
- 即使用户使用英文表达，也要用中文返回 summary 和 clarification_question。

参考旅行知识：
{knowledge_text or "无相关旅行知识。"}

用户消息：
{message}
""".strip()


def _response_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None

    candidates = [text]
    candidates.extend(match.group(0) for match in re.finditer(r"\{.*?\}", text, re.DOTALL))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    return None


def classify_with_llm(message: str, knowledge_text: str = "") -> TripChangeIntent:
    logger.info(
        "Classifying trip change intent with LLM: has_knowledge=%s",
        bool(knowledge_text.strip()),
    )
    prompt = build_intent_classification_prompt(
        message=message,
        knowledge_text=knowledge_text,
    )
    response = get_chat_model().invoke(prompt)
    response_text = _response_content(response)
    logger.debug("LLM intent raw response preview: %s", response_text[:500])
    data = _extract_json_object(response_text)
    if data is None:
        logger.warning("LLM intent response did not contain a JSON object")
        raise ValueError("LLM intent classifier did not return a JSON object")

    intent = TripChangeIntent(**data)
    logger.info("LLM intent classified: change_type=%s", intent.change_type)
    return intent


def classify_trip_change_intent(
    message: str,
    knowledge_text: str = "",
    use_llm: bool = False,
) -> TripChangeIntent:
    logger.info(
        "Classifying trip change intent: use_llm=%s has_knowledge=%s",
        use_llm,
        bool(knowledge_text.strip()),
    )
    if use_llm:
        try:
            return classify_with_llm(message, knowledge_text=knowledge_text)
        except Exception:
            logger.warning("LLM intent classification failed; falling back to rules", exc_info=True)
            intent = classify_with_rules(message)
            logger.info("Rule fallback intent classified: change_type=%s", intent.change_type)
            return intent

    intent = classify_with_rules(message)
    logger.info("Rule intent classified: change_type=%s", intent.change_type)
    return intent
