# LangChain 旅行规划项目迁移实施计划

> **给执行任务的 Agent 或开发者：** 如果后续要按这份计划实现代码，推荐使用 `superpowers:subagent-driven-development`，也可以使用 `superpowers:executing-plans`。每个步骤都用复选框标记，方便边做边勾。

**目标：** 将当前基于 HelloAgents 的旅行规划项目，逐步迁移为基于 LangChain + LangGraph + MCP + RAG 的多智能体旅行规划系统。

**总体架构：** 保留现有 FastAPI 后端、Vue 前端、Pydantic 数据模型和高德地图服务封装，先不要推倒重写。迁移重点放在“智能体编排层”：先做 LangGraph 规划流程，再加入旅行规划会话、对话式编辑、小改动即时应用、大改动确认后重新规划，最后接入 RAG 知识库。

**技术栈：** FastAPI、Pydantic v2、LangChain、LangGraph、langchain-openai、langchain-mcp-adapters、MCP 高德地图服务、向量检索/RAG、Vue 3、TypeScript、Ant Design Vue、pytest。

---

## 里程碑 0：理解当前项目基线

**目的：** 在迁移前先搞清楚当前项目是怎么跑通的。

**学习重点：** FastAPI 请求流程、Pydantic 数据结构、HelloAgents 当前编排方式、高德 MCP 当前接入方式。

**需要阅读的文件：**
- `backend/app/agents/trip_planner_agent.py`
- `backend/app/services/llm_service.py`
- `backend/app/services/amap_service.py`
- `backend/app/models/schemas.py`
- `backend/app/api/routes/trip.py`
- `frontend/src/services/api.ts`
- `frontend/src/views/Home.vue`
- `frontend/src/views/Result.vue`

- [ ] **步骤 1：梳理当前请求链路**

当前大致流程是：

```text
Home.vue
-> frontend API client
-> POST /api/trip/plan
-> backend/app/api/routes/trip.py
-> get_trip_planner_agent()
-> MultiAgentTripPlanner.plan_trip()
-> TripPlanResponse
```

- [ ] **步骤 2：运行当前测试**

执行：

```powershell
pytest tests -v
```

预期：现有测试通过，或者暴露出迁移前就存在的基线问题。

- [ ] **步骤 3：手动跑通当前应用**

启动后端和前端，在页面生成一份旅行计划。

预期：HelloAgents 旧版本仍然可用。迁移过程中不要一开始就破坏它。

---

## 里程碑 1：引入 LangChain 依赖和 LLM 适配器

**目的：** 先把 LangChain 引进项目，但暂时不改变用户功能。

**学习重点：** LangChain 的 Chat Model、环境变量配置、OpenAI 兼容接口。

**涉及文件：**
- 修改：`backend/requirements.txt`
- 修改：`backend/app/services/llm_service.py`
- 新增：`tests/test_langchain_llm_service.py`

- [ ] **步骤 1：添加依赖**

在 `backend/requirements.txt` 中添加：

```text
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langgraph>=0.2.0
langchain-mcp-adapters>=0.1.0
```

- [ ] **步骤 2：为 LangChain LLM 创建测试**

新增 `tests/test_langchain_llm_service.py`：

```python
from backend.app.services.llm_service import reset_llm


def test_langchain_llm_uses_openai_compatible_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    from backend.app.services.llm_service import get_chat_model

    reset_llm()
    model = get_chat_model()

    assert model.model_name == "test-model"
```

- [ ] **步骤 3：先运行测试，确认失败**

执行：

```powershell
pytest tests/test_langchain_llm_service.py -v
```

预期：失败，因为 `get_chat_model` 还没有实现。

- [ ] **步骤 4：实现 `get_chat_model`**

在 `backend/app/services/llm_service.py` 中保留旧的 `get_llm()`，同时新增：

```python
from langchain_openai import ChatOpenAI

_chat_model_instance = None


def get_chat_model() -> ChatOpenAI:
    global _chat_model_instance

    if _chat_model_instance is None:
        settings = get_settings()
        _chat_model_instance = ChatOpenAI(
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            temperature=0.3,
        )

    return _chat_model_instance
```

同时修改 `reset_llm()`，让它同时清空旧的 HelloAgents LLM 和新的 LangChain Chat Model 单例。

- [ ] **步骤 5：再次运行测试**

执行：

```powershell
pytest tests/test_langchain_llm_service.py -v
```

预期：测试通过。

---

## 里程碑 2：搭建 LangGraph 规划流程骨架

**目的：** 先搭出 LangGraph 工作流，但暂时不用真实 LLM 或真实 MCP，方便测试。

**学习重点：** LangGraph 的 State、Node、Edge、compile、invoke。

**涉及文件：**
- 新增：`backend/app/agents/langgraph_trip_planner.py`
- 新增：`tests/test_langgraph_trip_planner.py`

- [ ] **步骤 1：定义图状态**

新增 `backend/app/agents/langgraph_trip_planner.py`：

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from ..models.schemas import TripRequest, TripPlan


class TripPlannerState(TypedDict, total=False):
    request: TripRequest
    attractions_text: str
    weather_text: str
    hotels_text: str
    plan: TripPlan
    error: Optional[str]
```

- [ ] **步骤 2：写一个骨架测试**

新增 `tests/test_langgraph_trip_planner.py`：

```python
from backend.app.agents.langgraph_trip_planner import build_trip_planner_graph
from backend.app.models.schemas import TripRequest


def test_trip_planner_graph_returns_state():
    graph = build_trip_planner_graph()
    request = TripRequest(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )

    result = graph.invoke({"request": request})

    assert result["request"].city == "北京"
    assert "attractions_text" in result
    assert "weather_text" in result
    assert "hotels_text" in result
```

- [ ] **步骤 3：运行测试，确认失败**

执行：

```powershell
pytest tests/test_langgraph_trip_planner.py -v
```

预期：失败，因为 `build_trip_planner_graph` 还没有实现。

- [ ] **步骤 4：实现三个占位节点**

在 `langgraph_trip_planner.py` 中加入：

```python
def collect_attractions(state: TripPlannerState) -> TripPlannerState:
    request = state["request"]
    return {"attractions_text": f"{request.city} 景点候选"}


def collect_weather(state: TripPlannerState) -> TripPlannerState:
    request = state["request"]
    return {"weather_text": f"{request.city} 天气信息"}


def collect_hotels(state: TripPlannerState) -> TripPlannerState:
    request = state["request"]
    return {"hotels_text": f"{request.city} 酒店候选"}


def build_trip_planner_graph():
    builder = StateGraph(TripPlannerState)
    builder.add_node("collect_attractions", collect_attractions)
    builder.add_node("collect_weather", collect_weather)
    builder.add_node("collect_hotels", collect_hotels)
    builder.set_entry_point("collect_attractions")
    builder.add_edge("collect_attractions", "collect_weather")
    builder.add_edge("collect_weather", "collect_hotels")
    builder.add_edge("collect_hotels", END)
    return builder.compile()
```

- [ ] **步骤 5：再次运行测试**

执行：

```powershell
pytest tests/test_langgraph_trip_planner.py -v
```

预期：测试通过。

---

## 里程碑 3：在现有接口后面接入 LangGraph

**目的：** 保持 `POST /api/trip/plan` 不变，但可以通过配置切换到 LangGraph。

**学习重点：** 兼容式迁移、feature flag、保留旧功能作为兜底。

**涉及文件：**
- 修改：`backend/app/agents/langgraph_trip_planner.py`
- 修改：`backend/app/api/routes/trip.py`
- 修改：`backend/app/config.py`
- 新增：`tests/test_trip_route_langgraph.py`

- [ ] **步骤 1：创建 LangGraph 规划器外壳**

在 `backend/app/agents/langgraph_trip_planner.py` 中加入：

```python
class LangGraphTripPlanner:
    def __init__(self):
        self.graph = build_trip_planner_graph()

    def plan_trip(self, request: TripRequest) -> TripPlan:
        result = self.graph.invoke({"request": request})
        if "plan" not in result:
            return create_fallback_plan(request)
        return result["plan"]
```

- [ ] **步骤 2：添加基础兜底计划**

在同一文件中加入一个 `create_fallback_plan(request)`。它先生成一个简单但合法的 `TripPlan`，保证接口不会因为 LangGraph 还没完全接 LLM 就崩。

核心要求：

```text
城市、开始日期、结束日期来自 TripRequest
天数等于 request.travel_days
每天至少有一个景点
每天包含早饭、午饭、晚饭
返回值必须能通过 TripPlan 校验
```

- [ ] **步骤 3：添加配置开关**

在 `backend/app/config.py` 的 `Settings` 中新增：

```python
agent_backend: str = "helloagents"
```

含义：

```text
AGENT_BACKEND=helloagents 使用旧版 HelloAgents
AGENT_BACKEND=langgraph 使用新版 LangGraph
```

- [ ] **步骤 4：修改 `trip.py` 路由**

在 `backend/app/api/routes/trip.py` 中根据配置选择后端：

```python
from ...config import get_settings
from ...agents.langgraph_trip_planner import LangGraphTripPlanner

_langgraph_planner = None


def get_langgraph_trip_planner() -> LangGraphTripPlanner:
    global _langgraph_planner
    if _langgraph_planner is None:
        _langgraph_planner = LangGraphTripPlanner()
    return _langgraph_planner
```

在 `plan_trip` 中替换调用逻辑：

```python
settings = get_settings()
if settings.agent_backend.lower() == "langgraph":
    trip_plan = get_langgraph_trip_planner().plan_trip(request)
else:
    agent = get_trip_planner_agent()
    trip_plan = agent.plan_trip(request)
```

- [ ] **步骤 5：测试 LangGraph 分支**

执行：

```powershell
$env:AGENT_BACKEND="langgraph"
pytest tests -v
```

预期：LangGraph 分支不需要初始化 HelloAgents，也能返回合法旅行计划。

---

## 里程碑 4：用 LangChain MCP Adapters 接入高德 MCP

**目的：** 把当前 HelloAgents 的 MCP 调用方式，迁移到 LangChain 工具体系。

**学习重点：** MCP 协议、stdio MCP server、工具列表、工具调用、工具返回解析。

**涉及文件：**
- 新增：`backend/app/services/mcp_tools.py`
- 修改：`backend/app/agents/langgraph_trip_planner.py`
- 新增：`tests/test_mcp_tools.py`

- [ ] **步骤 1：创建 MCP 工具加载器**

新增 `backend/app/services/mcp_tools.py`：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from ..config import get_settings


def create_amap_mcp_client() -> MultiServerMCPClient:
    settings = get_settings()
    return MultiServerMCPClient(
        {
            "amap": {
                "command": "uvx",
                "args": ["amap-mcp-server"],
                "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key},
                "transport": "stdio",
            }
        }
    )
```

- [ ] **步骤 2：提供异步工具读取函数**

继续添加：

```python
async def get_amap_langchain_tools():
    client = create_amap_mcp_client()
    return await client.get_tools()
```

- [ ] **步骤 3：写测试，避免真的启动 MCP 服务**

测试时用 monkeypatch 替换 `MultiServerMCPClient`，只验证配置是否正确。

测试重点：

```text
配置里有 amap
command 是 uvx
args 包含 amap-mcp-server
env 里包含 AMAP_MAPS_API_KEY
```

- [ ] **步骤 4：把 MCP 工具接入 LangGraph 节点**

逐步让这些节点使用真实数据：

```text
collect_attractions -> maps_text_search
collect_weather -> maps_weather
collect_hotels -> maps_text_search
```

预期：MCP 成功时使用真实高德数据；失败时返回兜底文本，不影响整体生成。

---

## 里程碑 5：加入旅行规划会话 Trip Session

**目的：** 支持用户生成初稿后进入编辑对话，而不是一次请求结束。

**学习重点：** 有状态 API、会话模型、先用内存存储再考虑数据库。

**涉及文件：**
- 修改：`backend/app/models/schemas.py`
- 新增：`backend/app/services/trip_session_service.py`
- 修改：`backend/app/api/routes/trip.py`
- 新增：`tests/test_trip_session_service.py`

- [ ] **步骤 1：新增会话模型**

在 `schemas.py` 中添加：

```python
class ChatMessage(BaseModel):
    role: str = Field(..., description="user/assistant/system")
    content: str = Field(..., description="消息内容")


class TripSession(BaseModel):
    id: str
    request: TripRequest
    current_plan: TripPlan
    messages: List[ChatMessage] = Field(default_factory=list)
    status: str = "draft"


class TripSessionResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[TripSession] = None
```

- [ ] **步骤 2：实现内存会话服务**

新增 `trip_session_service.py`，先实现这些函数：

```text
create_session(request, plan)
get_session(session_id)
append_message(session_id, message)
update_plan(session_id, plan)
```

- [ ] **步骤 3：增加 API**

新增接口：

```text
POST /api/trip/sessions
GET /api/trip/sessions/{session_id}
```

预期：用户生成的行程可以被包装成一个 session，并且能通过 id 再取出来。

---

## 里程碑 6：实现混合编辑：小改动即时应用，大改动待确认

**目的：** 实现你选定的第三种交互模式。

**学习重点：** 意图分类、结构化输出、对 JSON 行程做确定性 patch。

**涉及文件：**
- 修改：`backend/app/models/schemas.py`
- 新增：`backend/app/agents/trip_edit_agent.py`
- 新增：`backend/app/services/trip_patch_service.py`
- 新增：`tests/test_trip_patch_service.py`

- [ ] **步骤 1：新增修改意图模型**

在 `schemas.py` 中添加：

```python
class TripChangeIntent(BaseModel):
    change_type: str = Field(..., description="small_change/major_revision/clarification_needed")
    summary: str
    patch_operations: List[dict] = Field(default_factory=list)
    clarification_question: Optional[str] = None


class TripChatRequest(BaseModel):
    message: str


class TripChatResponse(BaseModel):
    success: bool
    message: str = ""
    intent: Optional[TripChangeIntent] = None
    session: Optional[TripSession] = None
```

- [ ] **步骤 2：实现 PatchEngine**

第一版只支持四种小改动：

```text
remove_attraction：删除某天某个景点
replace_meal_text：修改某天某餐描述
update_day_description：修改某天行程描述
update_accommodation：修改住宿偏好
```

- [ ] **步骤 3：逐个写测试**

至少测试：

```text
删除第 1 天第 1 个景点，只影响这一天
修改晚餐描述，不影响早餐和午餐
错误 day_index 会抛出清晰异常
错误 attraction_index 会抛出清晰异常
```

- [ ] **步骤 4：新增聊天接口**

新增：

```text
POST /api/trip/sessions/{session_id}/chat
```

预期行为：

```text
small_change：立即应用 patch，右侧行程预览刷新
major_revision：只保存修改建议，不立即改当前行程
clarification_needed：追加一个助手追问
```

---

## 里程碑 7：实现大改动重新规划图

**目的：** 当用户提出会影响全局的需求时，重新调用规划流程生成新版行程。

**学习重点：** 人机确认、LangGraph 分支、revision prompt、计划校验。

**涉及文件：**
- 新增：`backend/app/agents/revision_planner_graph.py`
- 修改：`backend/app/api/routes/trip.py`
- 新增：`tests/test_revision_planner_graph.py`

- [ ] **步骤 1：定义重新规划状态**

状态中至少包含：

```text
session：当前旅行会话
revision_summary：用户确认的大改动摘要
original_plan：原行程
revised_plan：新行程
validation_errors：校验错误
```

- [ ] **步骤 2：新增重新规划接口**

新增：

```text
POST /api/trip/sessions/{session_id}/revise
```

预期：读取 session 中保存的大改动建议，调用重新规划图，返回新版 `TripPlan`。

- [ ] **步骤 3：保留旧版本**

在替换 `current_plan` 前，把旧版计划保存进 session 的历史版本列表中。

---

## 里程碑 8：加入 RAG 旅行知识库

**目的：** 让编辑对话智能体不只看当前行程，还能参考旅行知识。

**学习重点：** 文档加载、文本切分、向量化、向量检索、Hybrid RAG。

**涉及文件：**
- 新增：`backend/app/rag/travel_knowledge_loader.py`
- 新增：`backend/app/rag/vector_store.py`
- 新增：`backend/app/rag/travel_retriever.py`
- 新增：`backend/data/travel_knowledge/README.md`
- 修改：`backend/app/agents/trip_edit_agent.py`
- 新增：`tests/test_travel_retriever.py`

- [ ] **步骤 1：建立本地知识库目录**

新增 `backend/data/travel_knowledge/`，先放少量 markdown 文件：

```text
beijing_family.md：北京亲子旅行建议
beijing_food.md：北京美食主题建议
general_budget_tips.md：预算控制建议
general_low_walking_tips.md：少走路轻松游建议
```

- [ ] **步骤 2：实现文档加载**

读取 markdown 文件，并给每份文档加 metadata：

```python
{"source": file_name, "city": inferred_city_or_general}
```

- [ ] **步骤 3：实现检索器**

第一版测试可以先用 fake retriever，保证逻辑跑通；模型配置稳定后再接真实 embeddings 和向量库。

- [ ] **步骤 4：把检索结果注入编辑智能体**

预期：用户说“更适合亲子”“想吃更多当地美食”“少走路”时，对话智能体会先拿到相关知识，再判断是小改动还是大改动。

---

## 里程碑 9：前端加入混合编辑界面

**目的：** 用户点击“编辑”后，进入左侧聊天、右侧行程预览的界面。

**学习重点：** Vue 状态管理、API 类型定义、小改动即时刷新、大改动确认流程。

**涉及文件：**
- 修改：`frontend/src/types/index.ts`
- 修改：`frontend/src/services/api.ts`
- 新增：`frontend/src/views/EditSession.vue`
- 修改：`frontend/src/views/Result.vue`
- 修改：`frontend/src/App.vue` 或当前路由配置文件

- [ ] **步骤 1：新增 TypeScript 类型**

新增这些类型：

```text
TripSession
ChatMessage
TripChangeIntent
TripChatResponse
```

- [ ] **步骤 2：新增 API client 方法**

新增：

```text
createTripSession
getTripSession
sendTripChatMessage
applyTripRevision
confirmTripSession
```

- [ ] **步骤 3：新增编辑页路由**

编辑页布局：

```text
左侧：聊天消息 + 输入框
右侧：当前行程预览
底部或右上：应用重新规划 / 满意并完成
```

- [ ] **步骤 4：实现小改动即时刷新**

当后端返回 `small_change` 时，直接用 `response.session.current_plan` 刷新右侧预览。

- [ ] **步骤 5：实现大改动确认**

当后端返回 `major_revision` 时，先展示修改摘要，用户点确认后再调用 `/revise`。

---

## 里程碑 10：可观测性、校验和清理

**目的：** 让新架构更稳定、更容易学习和调试。

**学习重点：** 图节点日志、计划校验、移除旧依赖的时机。

**涉及文件：**
- 新增：`backend/app/services/trip_plan_validator.py`
- 修改：`backend/app/agents/langgraph_trip_planner.py`
- 修改：`backend/app/agents/revision_planner_graph.py`
- 修改：`README.md`

- [ ] **步骤 1：添加 TripPlan 校验器**

校验内容：

```text
日期范围是否匹配 request
天数是否等于 travel_days
景点是否重复
每个景点是否有坐标
每天是否包含 breakfast/lunch/dinner
如果有预算，budget.total 是否等于各部分合计
```

- [ ] **步骤 2：添加图节点调试日志**

每个 LangGraph 节点进入和退出时打印简要日志，但不要打印 API key。

- [ ] **步骤 3：更新 README**

说明：

```text
LangChain/LangGraph 新架构
MCP 工具连接方式
RAG 编辑智能体
混合编辑工作流
环境变量
如何运行测试
```

- [ ] **步骤 4：在功能对齐后移除 HelloAgents**

只有当下面都满足时才移除：

```text
AGENT_BACKEND=langgraph 能正常生成计划
编辑对话可用
大改动重新规划可用
测试通过
前端手动冒烟测试通过
```

---

## 推荐学习顺序

1. **LangChain Chat Model 基础**：理解 `ChatOpenAI`、messages、structured output。
2. **LangGraph 基础**：理解 state、node、edge、compile、invoke。
3. **MCP 基础**：理解 stdio MCP server、工具列表、工具调用。
4. **结构化输出**：让模型稳定输出 `TripPlan` 和 `TripChangeIntent`。
5. **RAG 基础**：加载文档、切分文档、向量化、检索。
6. **人机协作流程**：根据用户输入在“小改动 patch”和“大改动重新规划”之间分支。

## 推荐提交顺序

1. `chore: add langchain dependencies and chat model adapter`
2. `feat: add langgraph trip planner skeleton`
3. `feat: support langgraph planner behind feature flag`
4. `feat: add langchain mcp tool loader`
5. `feat: add trip planning sessions`
6. `feat: add trip edit classifier and patch engine`
7. `feat: add major trip revision graph`
8. `feat: add rag travel knowledge retriever`
9. `feat: add hybrid editing UI`
10. `docs: document langchain trip planner architecture`

## 一句话总结

这次迁移不要一口气重写。最稳的路线是：

```text
先让 LangChain 能调用模型
再让 LangGraph 能跑一个规划流程
再接 MCP 真实工具
再加 session
再加对话编辑
再加 RAG
最后改前端体验和清理旧框架
```

这样每一步都能运行、能测试、能学习，而且旧系统在前期一直可以作为兜底。
