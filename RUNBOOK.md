# HelloAgents Trip Planner 运行与验收手册

本文档用于本地启动、测试和验收旅行规划项目。

## 1. 项目目录

项目根目录：

```powershell
E:\agent\hello-agents-main\hello-agents-main\code\chapter13\helloagents-trip-planner
```

主要目录：

- `backend`：FastAPI 后端、LangGraph 规划器、RAG、Amap/Jina 服务
- `frontend`：Vue 前端
- `tests`：后端自动化测试
- `RUNBOOK.md`：当前运行手册

## 2. 环境配置

后端配置文件：

```powershell
backend\.env
```

至少需要配置：

```text
AMAP_API_KEY=你的高德地图Key
LLM_API_KEY=你的LLM API Key
LLM_BASE_URL=你的模型请求地址
LLM_MODEL_ID=你的模型ID
JINA_API_KEY=你的Jina API Key
JINA_EMBEDDING_MODEL=jina-embeddings-v4
JINA_EMBEDDING_TASK_QUERY=retrieval.query
JINA_EMBEDDING_TASK_DOCUMENT=retrieval.passage
```

如果访问 Jina API 需要代理，在当前 PowerShell 窗口设置：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
```

前端配置文件：

```powershell
frontend\.env
```

确认后端地址指向本地服务：

```text
VITE_API_BASE_URL=http://localhost:8000
```

## 3. 启动后端

在 PowerShell 中执行：

```powershell
cd E:\agent\hello-agents-main\hello-agents-main\code\chapter13\helloagents-trip-planner\backend
$env:PYTHONPATH = "."
$env:DEBUG = "false"
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
..\.venv\python.exe -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

看到下面内容表示后端启动成功：

```text
Application startup complete.
```

接口文档地址：

```text
http://localhost:8000/docs
```

## 4. 启动前端

新开一个 PowerShell 窗口：

```powershell
cd E:\agent\hello-agents-main\hello-agents-main\code\chapter13\helloagents-trip-planner\frontend
npm run dev
```

默认访问：

```text
http://localhost:5173
```

## 5. 自动化测试

在项目根目录执行：

```powershell
cd E:\agent\hello-agents-main\hello-agents-main\code\chapter13\helloagents-trip-planner
$env:PYTHONPATH = "."
$env:DEBUG = "false"
.\.venv\python.exe -m pytest tests -v --basetemp=.pytest-tmp
```

前端构建测试：

```powershell
cd E:\agent\hello-agents-main\hello-agents-main\code\chapter13\helloagents-trip-planner\frontend
npm run build
```

## 6. Jina Embedding 验证

在后端目录执行：

```powershell
cd E:\agent\hello-agents-main\hello-agents-main\code\chapter13\helloagents-trip-planner\backend
$env:PYTHONPATH = "."
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
..\.venv\python.exe scripts/check_embedding_service.py
```

正常结果应包含：

```text
dimensions=2048
```

## 7. RAG 检索验证

在项目根目录执行：

```powershell
cd E:\agent\hello-agents-main\hello-agents-main\code\chapter13\helloagents-trip-planner
$env:PYTHONPATH = "."
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
.\.venv\python.exe backend\scripts\check_travel_vector_retriever.py
```

这一步用于验证 Markdown 知识库、Jina 向量、向量检索和关键词兜底是否正常。

## 8. HTTP 会话与重规划验证

先启动后端，再在项目根目录执行：

```powershell
cd E:\agent\hello-agents-main\hello-agents-main\code\chapter13\helloagents-trip-planner
.\.venv\python.exe backend\scripts\check_trip_revision_http.py
```

这一步验证：

```text
创建 session -> 聊天识别整体调整 -> 写入 pending_revision_summary
```

完整重规划验证：

```powershell
.\.venv\python.exe backend\scripts\check_trip_revision_http.py --revise
```

这一步验证：

```text
创建 session -> 聊天识别整体调整 -> 确认重规划 -> 保存旧版本 -> 更新当前计划
```

正常结果应包含：

```text
"success": true
"message": "旅行计划已重新规划"
"pending_revision_summary": null
"plan_versions_count": 1
```

## 9. 前端手动验收流程

启动后端和前端后，在浏览器打开：

```text
http://localhost:5173
```

按下面流程验收：

1. 输入城市、日期、交通方式、住宿偏好、旅行偏好。
2. 点击生成旅行计划。
3. 确认页面能展示每日行程、景点、酒店、天气、建议。
4. 在结果页聊天框输入小修改，例如：`把第一天的故宫换成天坛`。
5. 确认计划局部更新，且不会重新生成整份计划。
6. 输入整体调整，例如：`带老人出行，少走路，节奏慢一点`。
7. 确认系统先提示是否重新规划。
8. 点击确认重规划。
9. 确认新计划更新，且建议中体现低步行、慢节奏、适合老人等约束。

## 10. 常见问题

### AMAP_API_KEY 未配置

确认 `backend\.env` 存在，并且包含：

```text
AMAP_API_KEY=你的高德地图Key
```

后端配置固定读取 `backend\.env`，不要只在项目根目录放 `.env`。

### Jina API 连接超时

先确认浏览器代理端口，例如当前使用：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
```

再验证：

```powershell
curl.exe https://api.jina.ai/v1/embeddings
```

如果返回 `AUTH_MISSING_API_KEY`，说明网络已经通了，只是缺少认证头，这是正常的连通性验证结果。

### 前端请求超时

优先检查：

1. 后端是否启动在 `http://localhost:8000`
2. `frontend\.env` 的 `VITE_API_BASE_URL` 是否正确
3. 后端日志是否出现 `POST /api/trip/plan HTTP/1.1 200 OK`
4. 浏览器控制台 Network 是否有 500、404 或超时请求

