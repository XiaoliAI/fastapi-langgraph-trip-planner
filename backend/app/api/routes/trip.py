"""旅行规划API路由"""
#前端请求 
 #   ↓
#FastAPI 路由层 (trip.py) ← 你在这里
  #  ↓
#业务服务层 (TripSessionService)
   # ↓
#数据层/Agent

from fastapi import APIRouter, HTTPException
from ...models.schemas import (
    TripRequest,
    TripPlanResponse,
    ErrorResponse
)
from ...agents.langgraph_trip_planner import LangGraphTripPlanner
from ...agents.revision_planner_graph import build_revision_planner_graph
from ...models.schemas import (
    TripRequest,
    TripPlan,
    TripPlanResponse,
    TripSessionResponse,
    ErrorResponse,
)
from ...services.trip_session_service import get_trip_session_service
from ...services.trip_experience_service import enrich_trip_plan_experience
from ...services.trip_route_optimizer import optimize_trip_attraction_routes
from ...models.schemas import (
    ChatMessage,
    TripRequest,
    TripPlanResponse,
    TripSessionCreateRequest,
    TripSessionResponse,
    TripChatRequest,
    TripChatResponse,
    TripChangeIntent,
    ErrorResponse,
)
from ...services.trip_plan_patch_service import (
    apply_trip_patch_operations,
    build_patch_result_from_message,
    build_patch_operations_from_message,
    build_patch_result_from_pending_intent,
)
from ...services.trip_change_intent_service import classify_trip_change_intent
from ...rag.travel_retriever import (
    retrieve_default_travel_knowledge_smart_text,
)
#LangGraphTripPlanner 创建成本高（需要加载模型、工具等）
#所有请求共享一个实例，节省资源
#避免重复初始化
router = APIRouter(prefix="/trip", tags=["旅行规划"])
_langgraph_planner = None
_revision_trip_planner = None
def get_langgraph_trip_planner() -> LangGraphTripPlanner:
    global _langgraph_planner

    if _langgraph_planner is None:
        _langgraph_planner = LangGraphTripPlanner()

    return _langgraph_planner

class RevisionTripPlanner:
    def __init__(self):
        self.graph = build_revision_planner_graph()

    async def revise_trip_result(self, session, revision_summary: str):
        return await self.graph.ainvoke(
            {
                "session": session,
                "revision_summary": revision_summary,
            }
        )

    async def revise_trip(self, session, revision_summary: str):
        result = await self.revise_trip_result(
            session=session,
            revision_summary=revision_summary,
        )
        revised_plan = result.get("revised_plan")
        if revised_plan is None:
            raise ValueError("Revision graph did not return revised_plan")
        return revised_plan

def get_revision_trip_planner() -> RevisionTripPlanner:
    global _revision_trip_planner

    if _revision_trip_planner is None:
        _revision_trip_planner = RevisionTripPlanner()

    return _revision_trip_planner

@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="生成旅行计划",
    description="根据用户输入的旅行需求,生成详细的旅行计划"
)
async def plan_trip(request: TripRequest):
    """
    生成旅行计划

    Args:
        request: 旅行请求参数

    Returns:
        旅行计划响应
    """
    try:
        print(f"\n{'='*60}")
        print(f"📥 收到旅行规划请求:")
        print(f"   城市: {request.city}")
        print(f"   日期: {request.start_date} - {request.end_date}")
        print(f"   天数: {request.travel_days}")
        print(f"{'='*60}\n")

        # 获取Agent实例
        print("🔄 获取 LangGraph 旅行规划器实例...")
        planner = get_langgraph_trip_planner()

        # 生成旅行计划
        print("🚀 开始通过 LangGraph 生成旅行计划...")
        trip_plan = await planner.plan_trip(request)
        trip_plan = optimize_trip_attraction_routes(trip_plan)
        trip_plan = await enrich_trip_plan_experience(trip_plan, request)

        print("✅ 旅行计划生成成功,准备返回响应\n")

        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功",
            data=trip_plan
        )

    except Exception as e:
        print(f"❌ 生成旅行计划失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"生成旅行计划失败: {str(e)}"
        )


@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常"
)
async def health_check():
    """健康检查"""
    try:
        planner = get_langgraph_trip_planner()

        return {
            "status": "healthy",
            "service": "trip-planner",
            "agent_name": planner.__class__.__name__,
            "tools_count": 0,
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"服务不可用: {str(e)}"
        )
#将旅行计划保存成session，供后续对话编辑使用
@router.post(
    "/sessions",
    response_model=TripSessionResponse,
    summary="创建旅行规划会话",
    description="保存一份旅行规划草稿，供后续对话编辑使用",
)
async def create_trip_session(payload: TripSessionCreateRequest):
    try:
        service = get_trip_session_service()
        session = service.create_session(
            request=payload.request,
            plan=payload.plan,
        )

        return TripSessionResponse(
            success=True,
            message="旅行规划会话创建成功",
            data=session,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"创建旅行规划会话失败: {str(e)}",
        )
#编辑页获取旅行计划
@router.get(
    "/sessions/{session_id}",
    response_model=TripSessionResponse,
    summary="获取旅行规划会话",
    description="根据会话ID获取当前旅行规划草稿",
)
async def get_trip_session(session_id: str):
    try:
        service = get_trip_session_service()
        session = service.get_session(session_id)

        return TripSessionResponse(
            success=True,
            message="旅行规划会话获取成功",
            data=session,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取旅行规划会话失败: {str(e)}",
        )

@router.post(
    "/sessions/{session_id}/chat",
    response_model=TripChatResponse,
    summary="旅行规划编辑对话",
    description="向指定旅行规划会话追加一轮编辑对话消息",
)
#保存用户消息
#识别用户修改意图
#保存助手回复并返回 intent
async def chat_with_trip_session(
    session_id: str,
    payload: TripChatRequest,
):
    try:
        service = get_trip_session_service()

        user_message = ChatMessage(
            role="user",
            content=payload.message,
        )

        session = service.append_message(session_id, user_message)

        if session.pending_patch_intent is not None:
            patch_result = await build_patch_result_from_pending_intent(
                message=payload.message,
                plan=session.current_plan,
                pending_patch_intent=session.pending_patch_intent,
            )

            operations = patch_result.operations
            intent = TripChangeIntent(
                change_type="small_change" if operations else "clarification_needed",
                summary="继续处理上一轮未完成的局部修改意图",
                patch_operations=operations,
                clarification_question=(
                    patch_result.pending_patch_intent.clarification_question
                    if patch_result.pending_patch_intent is not None
                    else None
                ),
            )

            if operations:
                updated_plan = await apply_trip_patch_operations(
                    plan=session.current_plan,
                    operations=operations,
                )
                session = service.update_plan(session_id, updated_plan)
                session = service.clear_pending_patch_intent(session_id)
                assistant_content = "已完成局部修改，当前行程已更新。"

            elif patch_result.pending_patch_intent is not None:
                session = service.update_pending_patch_intent(
                    session_id=session_id,
                    pending_patch_intent=patch_result.pending_patch_intent,
                )
                assistant_content = patch_result.pending_patch_intent.clarification_question

            else:
                assistant_content = "我还不能准确判断你想怎么修改。你可以继续补充一句，例如说明第几天或具体景点。"

            assistant_message = ChatMessage(
                role="assistant",
                content=assistant_content,
            )
            session = service.append_message(session_id, assistant_message)

            return TripChatResponse(
                success=True,
                message="编辑消息已保存",
                data=session,
                intent=intent,
            )

        knowledge_text = retrieve_default_travel_knowledge_smart_text(
            query=payload.message,
            city=session.request.city,
            limit=3,
        )

        intent = classify_trip_change_intent(
            message=payload.message,
            knowledge_text=knowledge_text,
            use_llm=True,
        )

        if intent.change_type == "small_change":
            patch_result = await build_patch_result_from_message(
                message=payload.message,
                plan=session.current_plan,
            )

            operations = patch_result.operations
            intent.patch_operations = operations

            if operations:
                updated_plan = await apply_trip_patch_operations(
                    plan=session.current_plan,
                    operations=operations,
                )
                session = service.update_plan(session_id, updated_plan)
                session = service.clear_pending_patch_intent(session_id)

                assistant_content = "已完成局部修改，当前行程已更新。"

            else:
                if patch_result.pending_patch_intent is not None:
                    session = service.update_pending_patch_intent(
                        session_id=session_id,
                        pending_patch_intent=patch_result.pending_patch_intent,
                    )
                    assistant_content = patch_result.pending_patch_intent.clarification_question
                else:
                    assistant_content = "我还不能准确判断你想怎么修改。你可以继续补充一句，例如说明第几天或具体景点。"

        elif intent.change_type == "major_revision":
            session = service.update_pending_revision_summary(
                session_id=session_id,
                revision_summary=intent.summary,
            )
            assistant_content = (
                f"已识别为整体调整需求：{intent.summary}。"
                "请确认是否基于当前行程重新规划。"
            )

        else:
            assistant_content = intent.clarification_question or "请补充你希望调整的具体内容。"

        assistant_message = ChatMessage(
            role="assistant",
            content=assistant_content,
        )
        session = service.append_message(session_id, assistant_message)

        return TripChatResponse(
            success=True,
            message="编辑消息已保存",
            data=session,
            intent=intent,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"旅行规划编辑对话失败: {str(e)}",
        )

@router.post(
    "/sessions/{session_id}/revise",
    response_model=TripSessionResponse,
    summary="确认并重新规划旅行会话",
    description="根据会话中保存的大改动摘要重新生成旅行计划",
)
async def revise_trip_session(session_id: str):
    try:
        service = get_trip_session_service()
        session = service.get_session(session_id)

        if not session.pending_revision_summary:
            raise ValueError("No pending revision summary found")

        planner = get_revision_trip_planner()
        revision_summary = session.pending_revision_summary

        if hasattr(planner, "revise_trip_result"):
            result = await planner.revise_trip_result(
                session=session,
                revision_summary=revision_summary,
            )
            validation_errors = result.get("validation_errors", [])

            if validation_errors:
                return TripSessionResponse(
                    success=False,
                    message=(
                        "重新规划结果未通过校验: "
                        + "；".join(validation_errors)
                    ),
                    data=session,
                )

            revised_plan = result.get("revised_plan")
            if revised_plan is None:
                raise ValueError("Revision graph did not return revised_plan")
        else:
            revised_plan = await planner.revise_trip(
                session=session,
                revision_summary=revision_summary,
            )

        revised_plan = optimize_trip_attraction_routes(revised_plan)
        revised_plan = await enrich_trip_plan_experience(
            revised_plan,
            session.request,
        )

        session = service.save_current_plan_version(session_id)
        session = service.update_plan(session_id, revised_plan)
        session = service.clear_pending_revision_summary(session_id)

        return TripSessionResponse(
            success=True,
            message="旅行计划已重新规划",
            data=session,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"重新规划旅行会话失败: {str(e)}",
        )
