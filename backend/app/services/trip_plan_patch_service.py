#小批量修改
#build_patch_operations_from_message()：把用户消息转成结构化操作。
#apply_trip_patch_operations()：把结构化操作应用到 TripPlan。
#parse_day_index()：识别“第一天 / 第1天”。
#find_attraction_name()：从当前计划里找用户想删的景点。
#model_copy(deep=True)：复制一份计划再改，避免直接修改原对象。


import json
import logging
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from ..models.schemas import Attraction, Location, TripPlan
from .llm_service import get_chat_model
from typing import Awaitable, Callable
from ..models.schemas import Attraction, Location, PendingPatchIntent, POIInfo, TripPlan
from .amap_langchain_service import search_pois
from dataclasses import dataclass
SearchPoisFunc = Callable[[str, str, int], Awaitable[List[POIInfo]]]
logger = logging.getLogger(__name__)
def build_plan_patch_prompt(message: str, plan: TripPlan) -> str:
    plan_summary = _build_plan_summary(plan)

    return f"""
你是旅行计划编辑指令解析器。

你的任务：根据用户消息和当前旅行计划，输出 JSON 格式的修改操作。

当前旅行计划：
{plan_summary}

支持的 operation 只有两种：

1. remove_attraction
删除某一天的某个景点：
{{
  "operation": "remove_attraction",
  "day_index": 1,
  "attraction_name": "故宫博物院"
}}

2. replace_attraction
替换某一天的某个景点：
{{
  "operation": "replace_attraction",
  "day_index": 1,
  "old_attraction_name": "故宫博物院",
  "new_attraction_name": "天坛公园"
}}

    3. 用户表达不明确时：
    [
      {{
        "operation": "clarification_needed",
        "known_fields": {{
          "operation": "replace_attraction",
          "new_target_preference": "适合拍照的地方"
        }},
        "missing_fields": ["day_index", "old_attraction_name"],
        "clarification_question": "你想修改第几天的哪个景点？"
      }}
    ]

要求：
- 只输出 JSON，不要输出解释。
- 输出格式必须是 JSON 数组。
- day_index 必须来自当前旅行计划。
- 删除或替换的旧景点名称必须使用当前旅行计划中的原始景点名称。
- 如果用户意图不清楚，输出空数组 []。
- 如果用户说“故宫”但计划里是“故宫博物院”，应该输出“故宫博物院”。

用户消息：
{message}
""".strip()


#先判断替换，再判断删除。因为“把 A 换成 B”也可能被误认为修改类小变动，但它不是删除操作。
@dataclass
class PatchBuildResult:
    operations: List[Dict[str, Any]]
    pending_patch_intent: Optional[PendingPatchIntent] = None

async def build_patch_operations_from_message(
    message: str,
    plan: TripPlan,
) -> List[Dict[str, Any]]:
    logger.info("Building patch operations from message")
    prompt = build_plan_patch_prompt(message, plan)
    chat_model = get_chat_model()
    response = await chat_model.ainvoke(prompt)

    content = getattr(response, "content", str(response))
    logger.debug("Patch parser raw response preview: %s", str(content)[:500])
    raw_operations = _parse_json_array(content)

    operations = _validate_patch_operations(raw_operations, plan)
    logger.info("Patch operations validated: count=%s", len(operations))
    return operations
async def build_patch_result_from_message(
    message: str,
    plan: TripPlan,
) -> PatchBuildResult:
    logger.info("Building patch result from message")
    prompt = build_plan_patch_prompt(message, plan)
    chat_model = get_chat_model()
    response = await chat_model.ainvoke(prompt)

    content = getattr(response, "content", str(response))
    logger.debug("Patch result raw response preview: %s", str(content)[:500])
    raw_operations = _parse_json_array(content)

    pending_patch_intent = _extract_pending_patch_intent(raw_operations)
    if pending_patch_intent is not None:
        logger.info(
            "Patch result needs clarification: operation=%s missing_fields=%s",
            pending_patch_intent.operation,
            pending_patch_intent.missing_fields,
        )
        return PatchBuildResult(
            operations=[],
            pending_patch_intent=pending_patch_intent,
        )

    operations = _validate_patch_operations(raw_operations, plan)
    logger.info("Patch result operations validated: count=%s", len(operations))
    return PatchBuildResult(
        operations=operations,
        pending_patch_intent=None,
    )

async def build_patch_result_from_pending_intent(
    message: str,
    plan: TripPlan,
    pending_patch_intent: PendingPatchIntent,
) -> PatchBuildResult:
    logger.info(
        "Building patch result from pending intent: operation=%s missing_fields=%s",
        pending_patch_intent.operation,
        pending_patch_intent.missing_fields,
    )
    prompt = build_pending_patch_prompt(
        message=message,
        plan=plan,
        pending_patch_intent=pending_patch_intent,
    )
    chat_model = get_chat_model()
    response = await chat_model.ainvoke(prompt)

    content = getattr(response, "content", str(response))
    logger.debug("Pending patch raw response preview: %s", str(content)[:500])
    raw_operations = _parse_json_array(content)

    next_pending_patch_intent = _extract_pending_patch_intent(raw_operations)
    if next_pending_patch_intent is not None:
        logger.info(
            "Pending patch still needs clarification: operation=%s missing_fields=%s",
            next_pending_patch_intent.operation,
            next_pending_patch_intent.missing_fields,
        )
        return PatchBuildResult(
            operations=[],
            pending_patch_intent=next_pending_patch_intent,
        )

    operations = _validate_patch_operations(raw_operations, plan)
    logger.info("Pending patch operations validated: count=%s", len(operations))
    return PatchBuildResult(
        operations=operations,
        pending_patch_intent=None,
    )

def build_pending_patch_prompt(
    message: str,
    plan: TripPlan,
    pending_patch_intent: PendingPatchIntent,
) -> str:
    plan_summary = _build_plan_summary(plan)
    pending_json = json.dumps(
        pending_patch_intent.model_dump(),
        ensure_ascii=False,
        indent=2,
    )

    return f"""
你是旅行计划编辑指令解析器。

当前有一个上一轮尚未补全的修改意图。你的任务是结合：
1. 上一轮已识别的信息
2. 用户本轮补充的信息
3. 当前旅行计划

输出 JSON 格式的修改操作。

当前旅行计划：
{plan_summary}

上一轮未完成的修改意图：
{pending_json}

如果信息已经完整，输出可执行操作：
[
  {{
    "operation": "replace_attraction",
    "day_index": 1,
    "old_attraction_name": "故宫博物院",
    "new_attraction_name": "天坛公园"
  }}
]

如果信息仍然不完整，继续输出 clarification_needed：
[
  {{
    "operation": "clarification_needed",
    "known_fields": {{
      "operation": "replace_attraction",
      "new_attraction_name": "天坛公园"
    }},
    "missing_fields": ["old_attraction_name"],
    "clarification_question": "你想替换哪个景点？"
  }}
]

要求：
- 只输出 JSON，不要输出解释。
- 输出格式必须是 JSON 数组。
- day_index 必须来自当前旅行计划。
- 删除或替换的旧景点名称必须使用当前旅行计划中的原始景点名称。
- 不要依赖固定关键词，要根据语义理解用户本轮补充的信息。

用户本轮消息：
{message}
""".strip()

async def apply_trip_patch_operations(
    plan: TripPlan,
    operations: List[Dict[str, Any]],
    search_pois_func: SearchPoisFunc = search_pois,
) -> TripPlan:
    logger.info("Applying trip patch operations: count=%s", len(operations))
    updated_plan = plan.model_copy(deep=True)

    for operation in operations:
        operation_name = operation.get("operation")

        if operation_name == "remove_attraction":
            _apply_remove_attraction(updated_plan, operation)
        elif operation_name == "replace_attraction":
            await _apply_replace_attraction(
                updated_plan,
                operation,
                search_pois_func,
            )

    return updated_plan

#build_patch_operations_from_message()：让 LLM 根据当前计划和用户消息生成结构化操作。
#validate_patch_operations()：防止 LLM 幻觉，比如输出不存在的 day_index 或不存在的旧景点。
#apply_trip_patch_operations()：只负责执行已经校验过的操作。

def _build_plan_summary(plan: TripPlan) -> str:
    lines = [
        f"城市：{plan.city}",
        f"日期：{plan.start_date} 至 {plan.end_date}",
        "每日景点：",
    ]

    for day in plan.days:
        attraction_names = [attraction.name for attraction in day.attractions]
        lines.append(f"- 第 {day.day_index} 天：{', '.join(attraction_names)}")

    return "\n".join(lines)


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not json_match:
        return []

    try:
        value = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return []

    if not isinstance(value, list):
        return []

    return [
        item
        for item in value
        if isinstance(item, dict)
    ]


def _validate_patch_operations(
    operations: List[Dict[str, Any]],
    plan: TripPlan,
) -> List[Dict[str, Any]]:
    validated: List[Dict[str, Any]] = []

    valid_day_indexes = {day.day_index for day in plan.days}

    for operation in operations:
        operation_name = operation.get("operation")
        day_index = operation.get("day_index")

        if day_index not in valid_day_indexes:
            continue

        if operation_name == "remove_attraction":
            attraction_name = operation.get("attraction_name")
            if _plan_has_attraction(plan, day_index, attraction_name):
                validated.append(
                    {
                        "operation": "remove_attraction",
                        "day_index": day_index,
                        "attraction_name": attraction_name,
                    }
                )

        elif operation_name == "replace_attraction":
            old_attraction_name = operation.get("old_attraction_name")
            new_attraction_name = operation.get("new_attraction_name")

            if (
                _plan_has_attraction(plan, day_index, old_attraction_name)
                and isinstance(new_attraction_name, str)
                and new_attraction_name.strip()
            ):
                validated.append(
                    {
                        "operation": "replace_attraction",
                        "day_index": day_index,
                        "old_attraction_name": old_attraction_name,
                        "new_attraction_name": new_attraction_name.strip(),
                    }
                )

    return validated


def _plan_has_attraction(
    plan: TripPlan,
    day_index: int,
    attraction_name: Optional[str],
) -> bool:
    if not isinstance(attraction_name, str):
        return False

    for day in plan.days:
        if day.day_index != day_index:
            continue

        return any(
            attraction.name == attraction_name
            for attraction in day.attractions
        )

    return False


def _apply_remove_attraction(
    plan: TripPlan,
    operation: Dict[str, Any],
) -> None:
    day_index = int(operation["day_index"])
    attraction_name = str(operation["attraction_name"])

    for day in plan.days:
        if day.day_index != day_index:
            continue

        day.attractions = [
            attraction
            for attraction in day.attractions
            if attraction.name != attraction_name
        ]
        return


async def _apply_replace_attraction(
    plan: TripPlan,
    operation: Dict[str, Any],
    search_pois_func: SearchPoisFunc,
) -> None:
    day_index = int(operation["day_index"])
    old_attraction_name = str(operation["old_attraction_name"])
    new_attraction_name = str(operation["new_attraction_name"])

    new_attraction = await _build_attraction_from_poi_search(
        attraction_name=new_attraction_name,
        city=plan.city,
        search_pois_func=search_pois_func,
    )

    for day in plan.days:
        if day.day_index != day_index:
            continue

        day.attractions = [
            new_attraction if attraction.name == old_attraction_name else attraction
            for attraction in day.attractions
        ]
        return
#优先用高德 POI 构造真实景点。
#如果高德没有结果，仍返回合法 Attraction，避免整个聊天接口失败。
async def _build_attraction_from_poi_search(
    attraction_name: str,
    city: str,
    search_pois_func: SearchPoisFunc,
) -> Attraction:
    pois = await search_pois_func(attraction_name, city, 5)

    if pois:
        poi = pois[0]
        logger.info(
            "Replacing attraction with Amap POI result: requested=%s selected=%s",
            attraction_name,
            poi.name,
        )
        return Attraction(
            name=poi.name,
            address=poi.address,
            location=poi.location,
            visit_duration=120,
            description=f"{poi.name}是根据用户修改需求通过高德地图搜索加入的景点。",
            category=poi.type or "景点",
            poi_id=poi.id,
            ticket_price=0,
        )

    logger.warning(
        "No Amap POI result for replacement; using fallback attraction: name=%s city=%s",
        attraction_name,
        city,
    )
    return Attraction(
        name=attraction_name,
        address=city,
        location=Location(longitude=0, latitude=0),
        visit_duration=120,
        description=f"{attraction_name}是根据用户修改需求替换加入的景点，但未能从高德地图获取详细信息。",
        category="景点",
        ticket_price=0,
    )
@dataclass
class PatchBuildResult:
    operations: List[Dict[str, Any]]
    pending_patch_intent: Optional[PendingPatchIntent] = None

def _extract_pending_patch_intent(
    operations: List[Dict[str, Any]],
) -> Optional[PendingPatchIntent]:
    for operation in operations:
        if operation.get("operation") != "clarification_needed":
            continue

        known_fields = operation.get("known_fields")
        missing_fields = operation.get("missing_fields")
        clarification_question = operation.get("clarification_question")

        if not isinstance(known_fields, dict):
            known_fields = {}

        if not isinstance(missing_fields, list):
            missing_fields = []

        if not isinstance(clarification_question, str) or not clarification_question.strip():
            clarification_question = "请补充你想调整的具体内容。"

        inferred_operation = known_fields.get("operation")
        if not isinstance(inferred_operation, str) or not inferred_operation.strip():
            inferred_operation = "unknown"

        return PendingPatchIntent(
            operation=inferred_operation,
            known_fields=known_fields,
            missing_fields=[str(item) for item in missing_fields],
            clarification_question=clarification_question.strip(),
        )

    return None
