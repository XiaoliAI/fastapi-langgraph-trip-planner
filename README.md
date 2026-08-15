# 智能旅行规划助手

一个结合 LangChain、LangGraph、RAG、真实地图 POI 和 Vue 前端的智能旅行规划应用，是对Hellow Agent旅行项目的二次开发。

用户输入城市、日期、交通方式、住宿偏好和旅行偏好后，系统可以生成多日旅行计划，并展示景点、地图、天气、餐饮和酒店候选。

## 功能特性

### 旅行规划

- 使用 LangGraph 编排旅行规划流程。
- 使用 LangChain 调用兼容 OpenAI API 的 LLM。
- 每天默认安排两个景点。
- 根据经纬度将距离较近的景点安排在同一天。
- 全行程景点去重。
- 地图编号与每日景点编号保持一致。

### 真实地图数据

- 高德 POI 景点搜索。
- 高德餐饮 POI 搜索。
- 高德酒店候选搜索。
- 高德天气查询。
- 地址地理编码。
- 景点、餐馆和酒店距离计算。

### 景点和图片

- 景点图片优先使用高德数据。
- 支持景点多图轮播。
- 支持点击放大和左右切换。
- 景点评点摘要、打卡点和游玩建议。
- 景点图片按 `category=scenic` 分类获取。

### 餐饮和酒店

- 每天生成早餐、午餐和晚餐。
- 餐馆靠近当天对应景点。
- 餐馆跨餐次、跨天去重。
- 餐饮推荐理由支持 LLM 简短摘要。
- 餐饮图片按 `category=meal` 分类获取。
- 酒店模块独立展示最多四个候选。
- 酒店按到规划景点的平均距离排序。
- 酒店图片按 `category=hotel` 分类获取。

### RAG 和会话编辑

- 使用项目内 Markdown 文件作为旅行知识库。
- 使用 Jina Embedding 进行语义检索。
- 向量检索失败时回退到关键词检索。
- 支持旅行 session。
- 支持 Chat 局部修改。
- 支持确认后整体重新规划。

## 技术栈

### 后端

- Python 3.11+
- FastAPI
- Pydantic Settings
- LangChain
- LangGraph
- 高德地图 API / MCP
- Jina Embedding API

### 前端

- Vue 3
- TypeScript
- Vite
- Ant Design Vue
- Axios
- 高德地图 JavaScript API

## 项目结构

```text
helloagents-trip-planner/
├── backend/
│   ├── app/
│   │   ├── agents/       # LangGraph 和旅行规划 Agent
│   │   ├── api/          # FastAPI 路由
│   │   ├── models/       # Pydantic 数据模型
│   │   ├── rag/          # Markdown、Embedding 和检索
│   │   └── services/     # 高德、LLM、会话和体验增强服务
│   ├── data/
│   │   └── travel_knowledge/  # Markdown 旅行知识
│   ├── scripts/           # 联调和检查脚本
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── services/      # 前端 API
│   │   ├── types/         # TypeScript 类型
│   │   └── views/         # Home 和 Result 页面
│   ├── package.json
│   └── vite.config.ts
├── tests/
├── docs/
├── PROJECT_SUMMARY.md
├── RUNBOOK.md
└── README.md
```

## 环境要求

- Python 3.11 或更高版本
- Node.js 18 或更高版本
- npm
- 高德地图 API Key
- 兼容 OpenAI API 的 LLM API Key
- Jina API Key（启用向量检索时需要）
- 可选：Unsplash API Key

## 配置后端

进入后端目录：

```powershell
cd backend
```

创建 `backend/.env`，至少配置：

```dotenv
AMAP_API_KEY=你的高德地图Key

LLM_API_KEY=你的LLM_API_Key
LLM_BASE_URL=https://api-inference.modelscope.cn/v1
LLM_MODEL_ID=Qwen/Qwen3.5-35B-A3B

JINA_API_KEY=你的Jina_API_Key
```

可选配置：

```dotenv
LLM_TIMEOUT=60
UNSPLASH_ACCESS_KEY=你的Unsplash_Key
UNSPLASH_SECRET_KEY=你的Unsplash_Secret
```

不要把真实 `.env` 文件提交到 GitHub。

## 安装后端依赖

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果当前机器已经有项目根目录 `.venv`，也可以直接使用已有环境。

## 启动后端

在 `backend` 目录执行：

```powershell
$env:PYTHONPATH = "."
..\.venv\python.exe -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

后端接口文档：

- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 配置和启动前端

进入前端目录：

```powershell
cd frontend
npm install
```

前端使用高德 Web JS API 时，可以创建 `frontend/.env.local`：

```dotenv
VITE_API_BASE_URL=http://localhost:8000
VITE_AMAP_WEB_JS_KEY=你的高德Web端JS_Key
```

启动开发服务器：

```powershell
npm run dev
```

打开：

```text
http://localhost:5173
```

## 网络代理说明

如果当前网络访问 Jina、模型服务或其他外部 API 不稳定，可以在 PowerShell 当前会话设置代理：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
```

代理端口需要根据本机代理软件实际配置调整。

## 测试和构建

运行后端测试：

```powershell
.\.venv\python.exe -m pytest tests -v
```

运行前端构建：

```powershell
cd frontend
npm run build
```

检查 Embedding 服务：

```powershell
cd backend
..\.venv\python.exe scripts/check_embedding_service.py
```

检查旅行规划和行程修改：

```powershell
cd 项目根目录
.\.venv\python.exe backend\scripts\check_trip_revision_http.py --revise
```

## RAG 说明

当前 RAG 知识来源为：

```text
backend/data/travel_knowledge/
```

知识以 Markdown 文件保存，系统可以先进行向量检索，向量服务不可用时回退到关键词检索。

当前阶段不强制使用数据库。后续当知识规模、更新频率或用户数量增加时，再考虑接入向量数据库和持久化存储。

## 当前边界

- 免费 LLM 的响应速度和输出格式可能不稳定。
- 高德图片数量和质量取决于具体 POI 数据。
- 外部 API 需要正确的 Key、网络和代理配置。
- Session 当前主要用于本地运行和前后端联调。
- 餐饮、景点和酒店的 LLM 摘要均保留本地兜底。

## 发布注意事项

以下文件和目录不应提交到 GitHub：

- `.env` 和 `.env.*`
- `.venv/`
- `backend/.venv/`
- `frontend/node_modules/`
- `frontend/dist/`
- `.pytest_cache/`
- `__pycache__/`
- `.idea/`
- 本地日志和临时目录

发布前请确认：

1. 没有将 API Key 写入源码、README 或测试输出。
2. 没有提交真实 `.env` 文件。
3. 已执行后端测试或至少完成核心接口验证。
4. 已执行 `npm run build`。
5. README 中的启动命令与当前目录结构一致。

## 项目文档

- [项目总结](PROJECT_SUMMARY.md)
- [运行手册](RUNBOOK.md)
- [接口文档](http://localhost:8000/docs)（启动后端后访问）

## 许可证

当前许可证信息请以仓库最终发布版本中的 LICENSE 文件为准。发布到 GitHub 前建议补充明确的许可证文件。
