# LangChain Trip Planner Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the travel planner from HelloAgents to a LangChain/LangGraph architecture with MCP tools, hybrid human-in-the-loop revision, and a RAG-powered editing chat agent.

**Architecture:** Keep the existing FastAPI, Vue, Pydantic schemas, and Amap service boundaries while replacing the agent orchestration layer incrementally. Build a LangGraph planning workflow first, then add session state, patch-based editing, major revision flow, and finally RAG-backed travel chat.

**Tech Stack:** FastAPI, Pydantic v2, LangChain, LangGraph, langchain-openai-compatible chat model setup, langchain-mcp-adapters, MCP Amap server, vector store for RAG, Vue 3, TypeScript, Ant Design Vue, pytest.

---

## Milestone 0: Baseline And Learning Map

**Purpose:** Understand the current project before replacing anything.

**Learning Focus:** FastAPI route flow, Pydantic models, existing HelloAgents orchestration, current MCP usage.

**Files:**
- Read: `backend/app/agents/trip_planner_agent.py`
- Read: `backend/app/services/llm_service.py`
- Read: `backend/app/services/amap_service.py`
- Read: `backend/app/models/schemas.py`
- Read: `backend/app/api/routes/trip.py`
- Read: `frontend/src/services/api.ts`
- Read: `frontend/src/views/Home.vue`
- Read: `frontend/src/views/Result.vue`

- [ ] **Step 1: Document current request flow**

Create a short local note while studying:

```text
Home.vue -> frontend API client -> POST /api/trip/plan -> trip.py -> get_trip_planner_agent() -> MultiAgentTripPlanner.plan_trip() -> TripPlanResponse
```

- [ ] **Step 2: Run current backend tests**

Run:

```powershell
pytest tests -v
```

Expected: existing tests either pass or reveal baseline failures unrelated to the migration.

- [ ] **Step 3: Run a current app smoke test**

Start backend and frontend as usual, then generate a short trip plan from the UI.

Expected: the current HelloAgents version still works before migration begins.

---

## Milestone 1: Add LangChain Dependencies And LLM Adapter

**Purpose:** Introduce LangChain without changing product behavior.

**Learning Focus:** Chat model abstraction, environment-driven model config, LangChain message API.

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/services/llm_service.py`
- Create: `tests/test_langchain_llm_service.py`

- [ ] **Step 1: Add dependencies**

Add these packages to `backend/requirements.txt`:

```text
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langgraph>=0.2.0
langchain-mcp-adapters>=0.1.0
```

- [ ] **Step 2: Write a test for LangChain LLM construction**

Create `tests/test_langchain_llm_service.py`:

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

- [ ] **Step 3: Run the failing test**

Run:

```powershell
pytest tests/test_langchain_llm_service.py -v
```

Expected: FAIL because `get_chat_model` does not exist.

- [ ] **Step 4: Implement `get_chat_model` alongside the old LLM**

Modify `backend/app/services/llm_service.py` to keep `get_llm()` temporarily and add:

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

Update `reset_llm()` so it clears both singleton instances.

- [ ] **Step 5: Run tests**

Run:

```powershell
pytest tests/test_langchain_llm_service.py -v
```

Expected: PASS.

---

## Milestone 2: Build A LangGraph Planning Skeleton

**Purpose:** Create a new planning workflow that can be tested without calling real LLMs or MCP tools.

**Learning Focus:** LangGraph state, nodes, edges, deterministic workflow testing.

**Files:**
- Create: `backend/app/agents/langgraph_trip_planner.py`
- Create: `tests/test_langgraph_trip_planner.py`

- [ ] **Step 1: Define graph state**

Create `backend/app/agents/langgraph_trip_planner.py` with:

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

- [ ] **Step 2: Write a skeleton graph test**

Create `tests/test_langgraph_trip_planner.py`:

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

- [ ] **Step 3: Run the failing test**

Run:

```powershell
pytest tests/test_langgraph_trip_planner.py -v
```

Expected: FAIL because `build_trip_planner_graph` does not exist.

- [ ] **Step 4: Implement deterministic placeholder nodes**

Add:

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

- [ ] **Step 5: Run tests**

Run:

```powershell
pytest tests/test_langgraph_trip_planner.py -v
```

Expected: PASS.

---

## Milestone 3: Replace HelloAgents Planning Behind The Existing API

**Purpose:** Preserve `POST /api/trip/plan` while switching orchestration to LangGraph.

**Learning Focus:** Structured output, compatibility wrappers, incremental migration.

**Files:**
- Modify: `backend/app/agents/langgraph_trip_planner.py`
- Modify: `backend/app/api/routes/trip.py`
- Create: `tests/test_trip_route_langgraph.py`

- [ ] **Step 1: Add a LangGraph planner facade**

In `backend/app/agents/langgraph_trip_planner.py`, add:

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

- [ ] **Step 2: Add a minimal fallback plan function**

Add:

```python
from datetime import datetime, timedelta
from ..models.schemas import DayPlan, Attraction, Meal, Location


def create_fallback_plan(request: TripRequest) -> TripPlan:
    start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
    days = []
    for index in range(request.travel_days):
        current_date = start_date + timedelta(days=index)
        days.append(
            DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=index,
                description=f"第{index + 1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}推荐景点{index + 1}",
                        address=f"{request.city}",
                        location=Location(longitude=116.397128, latitude=39.916527),
                        visit_duration=120,
                        description="根据用户偏好生成的候选景点。",
                        category="景点",
                    )
                ],
                meals=[
                    Meal(type="breakfast", name="早餐", description="当地早餐"),
                    Meal(type="lunch", name="午餐", description="当地午餐"),
                    Meal(type="dinner", name="晚餐", description="当地晚餐"),
                ],
            )
        )

    return TripPlan(
        city=request.city,
        start_date=request.start_date,
        end_date=request.end_date,
        days=days,
        weather_info=[],
        overall_suggestions="这是 LangGraph 规划器生成的基础行程。",
    )
```

- [ ] **Step 3: Update route to use the new facade behind a feature flag**

In `backend/app/config.py`, add:

```python
agent_backend: str = "helloagents"
```

In `backend/app/api/routes/trip.py`, choose implementation:

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

Inside `plan_trip`, use:

```python
settings = get_settings()
if settings.agent_backend.lower() == "langgraph":
    trip_plan = get_langgraph_trip_planner().plan_trip(request)
else:
    agent = get_trip_planner_agent()
    trip_plan = agent.plan_trip(request)
```

- [ ] **Step 4: Test with `AGENT_BACKEND=langgraph`**

Run:

```powershell
$env:AGENT_BACKEND="langgraph"
pytest tests -v
```

Expected: API tests pass and no HelloAgents initialization is required for the LangGraph path.

---

## Milestone 4: Integrate MCP Tools Through LangChain

**Purpose:** Replace HelloAgents MCP tool calls with LangChain-compatible MCP tools.

**Learning Focus:** MCP protocol, `langchain-mcp-adapters`, tool invocation, tool result parsing.

**Files:**
- Create: `backend/app/services/mcp_tools.py`
- Modify: `backend/app/agents/langgraph_trip_planner.py`
- Create: `tests/test_mcp_tools.py`

- [ ] **Step 1: Create MCP tool loader**

Create `backend/app/services/mcp_tools.py`:

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

- [ ] **Step 2: Add an async tool accessor**

Add:

```python
async def get_amap_langchain_tools():
    client = create_amap_mcp_client()
    return await client.get_tools()
```

- [ ] **Step 3: Add tests with monkeypatched client**

Create `tests/test_mcp_tools.py`:

```python
from backend.app.services import mcp_tools


class FakeClient:
    async def get_tools(self):
        return ["maps_text_search", "maps_weather"]


def test_create_amap_mcp_client_has_amap_config(monkeypatch):
    captured = {}

    class CapturingClient:
        def __init__(self, config):
            captured.update(config)

    monkeypatch.setattr(mcp_tools, "MultiServerMCPClient", CapturingClient)
    monkeypatch.setenv("AMAP_API_KEY", "test-amap")

    mcp_tools.create_amap_mcp_client()

    assert "amap" in captured
    assert captured["amap"]["command"] == "uvx"
```

- [ ] **Step 4: Wire tool usage into graph nodes**

Update `collect_attractions`, `collect_weather`, and `collect_hotels` to accept service functions so tests can inject fake data before using real MCP.

Expected behavior: graph nodes produce text from real Amap data when available and fallback text when tools fail.

---

## Milestone 5: Add Trip Sessions

**Purpose:** Support edit conversations by storing current draft, messages, revision proposals, and final plan.

**Learning Focus:** Stateful API design, session models, in-memory repository before database.

**Files:**
- Modify: `backend/app/models/schemas.py`
- Create: `backend/app/services/trip_session_service.py`
- Modify: `backend/app/api/routes/trip.py`
- Create: `tests/test_trip_session_service.py`

- [ ] **Step 1: Add session models**

Add to `schemas.py`:

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

- [ ] **Step 2: Create in-memory session service**

Create `trip_session_service.py` with `create_session`, `get_session`, `append_message`, and `update_plan`.

- [ ] **Step 3: Add API endpoint**

Add:

```text
POST /api/trip/sessions
GET /api/trip/sessions/{session_id}
```

Expected: a generated plan can be wrapped into a session and fetched by id.

---

## Milestone 6: Build The Hybrid Edit Classifier And Patch Engine

**Purpose:** Implement the third interaction mode: small changes apply immediately, major changes require confirmation.

**Learning Focus:** Intent classification, structured outputs, deterministic patching.

**Files:**
- Modify: `backend/app/models/schemas.py`
- Create: `backend/app/agents/trip_edit_agent.py`
- Create: `backend/app/services/trip_patch_service.py`
- Create: `tests/test_trip_patch_service.py`

- [ ] **Step 1: Add revision models**

Add:

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

- [ ] **Step 2: Implement patch operations**

Support these first:

```text
remove_attraction
replace_meal_text
update_day_description
update_accommodation
```

- [ ] **Step 3: Test one patch at a time**

Write tests proving that removing an attraction changes only the requested day and that invalid indexes raise a clear `ValueError`.

- [ ] **Step 4: Add chat endpoint**

Add:

```text
POST /api/trip/sessions/{session_id}/chat
```

Expected:
- `small_change` applies patch immediately.
- `major_revision` stores a proposal but does not mutate the current plan.
- `clarification_needed` appends an assistant question.

---

## Milestone 7: Add Major Revision Planner

**Purpose:** Re-run planning when the requested change affects the whole itinerary.

**Learning Focus:** Human-in-the-loop graph branching, revision prompts, plan validation.

**Files:**
- Create: `backend/app/agents/revision_planner_graph.py`
- Modify: `backend/app/api/routes/trip.py`
- Create: `tests/test_revision_planner_graph.py`

- [ ] **Step 1: Create revision graph state**

State should include:

```text
session
revision_summary
original_plan
revised_plan
validation_errors
```

- [ ] **Step 2: Add endpoint**

Add:

```text
POST /api/trip/sessions/{session_id}/revise
```

Expected: uses the stored major revision proposal and returns a new `TripPlan`.

- [ ] **Step 3: Preserve old version**

Store previous plan versions in the session service before replacing `current_plan`.

---

## Milestone 8: Add RAG Knowledge Base For Editing Chat

**Purpose:** Let the edit agent answer and revise using travel knowledge beyond the current plan.

**Learning Focus:** Documents, embeddings, vector store, retrieval chains, hybrid RAG.

**Files:**
- Create: `backend/app/rag/travel_knowledge_loader.py`
- Create: `backend/app/rag/vector_store.py`
- Create: `backend/app/rag/travel_retriever.py`
- Create: `backend/data/travel_knowledge/README.md`
- Modify: `backend/app/agents/trip_edit_agent.py`
- Create: `tests/test_travel_retriever.py`

- [ ] **Step 1: Add a small local knowledge corpus**

Create markdown docs under `backend/data/travel_knowledge/`, starting with:

```text
beijing_family.md
beijing_food.md
general_budget_tips.md
general_low_walking_tips.md
```

- [ ] **Step 2: Implement document loading**

Load markdown files with metadata:

```python
{"source": file_name, "city": inferred_city_or_general}
```

- [ ] **Step 3: Implement retriever**

Start with an in-memory vector store or simple fake retriever in tests, then switch to real embeddings once model configuration is stable.

- [ ] **Step 4: Inject retrieved context into `TripEditAgent`**

Expected: when the user asks for family-friendly or food-focused changes, the edit agent sees relevant context before classifying the change.

---

## Milestone 9: Frontend Hybrid Editing Experience

**Purpose:** Add the left chat/right preview editing screen.

**Learning Focus:** Vue state management, API typing, optimistic UI for small patches, confirmation flow for major changes.

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/services/api.ts`
- Create: `frontend/src/views/EditSession.vue`
- Modify: `frontend/src/views/Result.vue`
- Modify: `frontend/src/App.vue` or router setup file

- [ ] **Step 1: Add TypeScript types**

Add types matching:

```text
TripSession
ChatMessage
TripChangeIntent
TripChatResponse
```

- [ ] **Step 2: Add API client methods**

Add:

```text
createTripSession
getTripSession
sendTripChatMessage
applyTripRevision
confirmTripSession
```

- [ ] **Step 3: Add edit route**

The edit view should show:

```text
left: chat messages and input
right: current trip plan preview
bottom/right actions: apply revision, confirm final plan
```

- [ ] **Step 4: Implement small-change refresh**

Expected: after a `small_change` response, update the right preview from `response.session.current_plan`.

- [ ] **Step 5: Implement major-change confirmation**

Expected: after `major_revision`, show the proposal summary and an explicit confirmation button before calling `/revise`.

---

## Milestone 10: Observability, Validation, And Cleanup

**Purpose:** Make the new stack easier to learn from and debug.

**Learning Focus:** graph tracing, validation boundaries, removing legacy code safely.

**Files:**
- Create: `backend/app/services/trip_plan_validator.py`
- Modify: `backend/app/agents/langgraph_trip_planner.py`
- Modify: `backend/app/agents/revision_planner_graph.py`
- Modify: `README.md`

- [ ] **Step 1: Add validator**

Validate:

```text
dates match request
number of days matches travel_days
no duplicate attractions
each attraction has a location
meals include breakfast/lunch/dinner
budget total matches parts when present
```

- [ ] **Step 2: Add graph debug logging**

Log each graph node entry and output summary without printing secrets.

- [ ] **Step 3: Update README**

Document:

```text
LangChain/LangGraph architecture
MCP tool connection
RAG edit agent
hybrid edit workflow
environment variables
how to run tests
```

- [ ] **Step 4: Remove HelloAgents dependency after parity**

Only remove `hello-agents` once:

```text
AGENT_BACKEND=langgraph works for normal planning
edit chat works
major revision works
tests pass
manual UI smoke test passes
```

---

## Suggested Learning Order

1. LangChain chat model basics: understand `ChatOpenAI`, messages, structured output.
2. LangGraph basics: understand state, node, edge, compile, invoke.
3. MCP basics: understand stdio MCP server, tool listing, tool invocation.
4. Structured output: make the model return `TripPlan` and `TripChangeIntent`.
5. RAG basics: load docs, split docs, embed docs, retrieve docs.
6. Human-in-the-loop workflows: branch between small patch and major revision.

## Recommended Commit Boundaries

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

## Self-Review

- Spec coverage: The plan covers LangChain migration, LangGraph multi-agent planning, MCP integration, RAG chat editing, hybrid small/major change flow, session state, frontend integration, validation, and cleanup.
- Placeholder scan: No `TBD` or vague implementation-only steps remain. Later milestones intentionally define behavior and files but leave exact code smaller than early migration steps because they depend on decisions validated by earlier tasks.
- Type consistency: `TripSession`, `ChatMessage`, `TripChangeIntent`, `TripChatRequest`, and `TripChatResponse` are introduced before use in API and frontend milestones.
