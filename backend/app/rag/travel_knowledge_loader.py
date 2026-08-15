#backend/app/rag/
#├── travel_knowledge_loader.py    # 加载 markdown
#├── travel_retriever.py           # 检索相关知识
#└── vector_store.py               # 后面如果接向量库再放这里

#遍历目录下所有 .md 文件
#读取文本
#加 metadata
#返回 TravelKnowledgeDocument 列表

from dataclasses import dataclass
from pathlib import Path
from typing import Any

#作用：定义 RAG 文档的统一格式
@dataclass
class TravelKnowledgeDocument:
    content: str
    metadata: dict[str, Any]

#作用：从文件名推断城市标签。
def infer_city_from_file_name(file_name: str) -> str:
    if file_name.startswith("beijing_"):
        return "北京"

    if file_name.startswith("general_"):
        return "general"

    return "unknown"

#作用：从文件名推断知识类型。
def infer_category_from_file_name(file_name: str) -> str:
    stem = Path(file_name).stem

    if "_" not in stem:
        return "general"

    parts = stem.split("_", 1)
    return parts[1]


def load_travel_knowledge_documents(
    knowledge_dir: str | Path,
) -> list[TravelKnowledgeDocument]:
    directory = Path(knowledge_dir)

    if not directory.exists() or not directory.is_dir():
        return []

    documents: list[TravelKnowledgeDocument] = []

    for markdown_file in sorted(directory.glob("*.md")):
        content = markdown_file.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append(
            TravelKnowledgeDocument(
                content=content,
                metadata={
                    "source": markdown_file.name,
                    "city": infer_city_from_file_name(markdown_file.name),
                    "category": infer_category_from_file_name(markdown_file.name),
                },
            )
        )

    return documents
#设置默认的知识目录，返回 TravelKnowledgeDocument 列表
def get_default_travel_knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "travel_knowledge"


def load_default_travel_knowledge_documents() -> list[TravelKnowledgeDocument]:
    return load_travel_knowledge_documents(get_default_travel_knowledge_dir())