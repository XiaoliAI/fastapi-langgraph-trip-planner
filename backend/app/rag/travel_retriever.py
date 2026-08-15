"""我们先定义想要的检索行为：
用户说“少走路”
应命中 general_low_walking_tips.md

用户说“本地美食”
应命中 beijing_food.md

用户说无关内容
返回空列表

检索结果能格式化成 prompt 文本

后面 LLM prompt 会这样用：
retrieved_text = format_retrieved_knowledge(results)
然后塞进：
参考旅行知识：
{retrieved_text}

加载 Markdown 知识
→ query 调 Jina retrieval.query
→ document 调 Jina retrieval.passage
→ cosine similarity 排序
→ 格式化成 prompt 文本

"""


"""用户 query 命中文档 category 的关键词 -> 加分
query 词出现在文档正文里 -> 加分
城市匹配 -> 加分
general 通用文档 -> 小加分"""

from .travel_knowledge_loader import (
    TravelKnowledgeDocument,
    load_default_travel_knowledge_documents,
)
from .embedding_service import embed_query, embed_document
from .vector_retriever import (
    retrieve_travel_knowledge_by_vector,
    retrieve_travel_knowledge_from_vector_documents,
)
from .vector_store import get_vector_knowledge_documents

QUERY_KEYWORDS = {
    "low_walking_tips": [
        "少走路",
        "轻松",
        "别太累",
        "老人",
        "儿童",
        "体力",
        "减少跨区域",
    ],
    "food": [
        "美食",
        "吃",
        "本地",
        "特色",
        "餐厅",
        "烤鸭",
        "炸酱面",
        "涮羊肉",
    ],
    "budget_tips": [
        "预算",
        "省钱",
        "便宜",
        "低价",
        "免费",
        "性价比",
    ],
    "family": [
        "亲子",
        "孩子",
        "儿童",
        "家庭",
        "科技馆",
        "动物园",
    ],
}


def retrieve_travel_knowledge(
    query: str,
    documents: list[TravelKnowledgeDocument],
    city: str = "",
    limit: int = 3,
) -> list[TravelKnowledgeDocument]:
    scored_documents: list[tuple[int, TravelKnowledgeDocument]] = []

    for document in documents:
        score = _score_document(
            query=query,
            document=document,
            city=city,
        )

        if score > 0:
            scored_documents.append((score, document))

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        document
        for _, document in scored_documents[:limit]
    ]


def _score_document(
    query: str,
    document: TravelKnowledgeDocument,
    city: str,
) -> int:
    relevance_score = 0
    boost_score = 0

    category = document.metadata.get("category", "")
    document_city = document.metadata.get("city", "")

    for keyword in QUERY_KEYWORDS.get(category, []):
        if keyword in query:
            relevance_score += 3

    for word in _tokenize_query(query):
        if word and word in document.content:
            relevance_score += 1

    if relevance_score == 0:
        return 0

    if city and document_city == city:
        boost_score += 2

    if document_city == "general":
        boost_score += 1

    return relevance_score + boost_score

def _tokenize_query(query: str) -> list[str]:
    separators = ["，", "。", "、", ",", ".", " ", "！", "？", "!", "?"]

    tokens = [query]

    for separator in separators:
        next_tokens: list[str] = []
        for token in tokens:
            next_tokens.extend(token.split(separator))
        tokens = next_tokens

    return [
        token.strip()
        for token in tokens
        if token.strip()
    ]


def format_retrieved_knowledge(
    documents: list[TravelKnowledgeDocument],
) -> str:
    if not documents:
        return "无相关旅行知识。"

    blocks = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        city = document.metadata.get("city", "unknown")
        category = document.metadata.get("category", "unknown")

        blocks.append(
            f"{index}. 来源: {source} | 城市: {city} | 类型: {category}\n"
            f"{document.content}"
        )

    return "\n\n".join(blocks)

"""它是 RAG 检索的业务入口。
以后在编辑智能体或重规划图里可以直接写：
knowledge_text = retrieve_default_travel_knowledge_text(
    query=user_message,
    city=session.request.city,
)"""

#向量检索入口
def retrieve_default_travel_knowledge_text(
    query: str,
    city: str = "",
    limit: int = 3,
) -> str:
    documents = load_default_travel_knowledge_documents()

    retrieved_documents = retrieve_travel_knowledge(
        query=query,
        documents=documents,
        city=city,
        limit=limit,
    )

    return format_retrieved_knowledge(retrieved_documents)

def retrieve_default_travel_knowledge_by_vector_text(
    query: str,
    city: str = "",
    limit: int = 3,
) -> str:
    documents = load_default_travel_knowledge_documents()

    retrieved_documents = retrieve_travel_knowledge_by_vector(
        query=query,
        documents=documents,
        embed_query=embed_query,
        embed_document=embed_document,
        city=city,
        limit=limit,
    )

    return format_retrieved_knowledge(retrieved_documents)

"""Jina API 正常：使用向量检索。
Jina API 报错：自动回退关键词检索。
向量检索没有结果：也回退关键词检索。
原来的两个检索函数都保留不动。"""

def retrieve_default_travel_knowledge_smart_text(
    query: str,
    city: str = "",
    limit: int = 3,
) -> str:
    """优先使用向量检索，失败或无结果时回退关键词检索。"""
    documents = load_default_travel_knowledge_documents()

    try:
        cached_vector_documents = get_vector_knowledge_documents(
            documents=documents,
            embed_document=embed_document,
        )
        vector_documents = retrieve_travel_knowledge_from_vector_documents(
            query=query,
            vector_documents=cached_vector_documents,
            embed_query=embed_query,
            city=city,
            limit=limit,
        )

        if vector_documents:
            return format_retrieved_knowledge(vector_documents)

    except Exception as exc:
        print(f"向量检索失败，回退关键词检索: {exc}")

    keyword_documents = retrieve_travel_knowledge(
        query=query,
        documents=documents,
        city=city,
        limit=limit,
    )

    return format_retrieved_knowledge(keyword_documents)
