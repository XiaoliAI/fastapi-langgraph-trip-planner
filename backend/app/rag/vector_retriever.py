"""1. query 转向量
2. document 转向量
3. 用 cosine similarity 排序
负责向量相似检索
"""
import math
from typing import Callable

from .travel_knowledge_loader import TravelKnowledgeDocument
from .vector_store import VectorKnowledgeDocument


EmbedTextFunc = Callable[[str], list[float]]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def retrieve_travel_knowledge_by_vector(
    query: str,
    documents: list[TravelKnowledgeDocument],
    embed_query: EmbedTextFunc,
    embed_document: EmbedTextFunc,
    city: str = "",
    limit: int = 3,
    min_score: float = 0.5,
) -> list[TravelKnowledgeDocument]:
    if not query.strip() or not documents:
        return []

    query_vector = embed_query(query)
    scored_documents: list[tuple[float, TravelKnowledgeDocument]] = []

    for document in documents:
        document_vector = embed_document(document.content)
        score = cosine_similarity(query_vector, document_vector)

        document_city = str(document.metadata.get("city", ""))
        if city and document_city == city:
            score += 0.05

        if score >= min_score:
            scored_documents.append((score, document))

    scored_documents.sort(key=lambda item: item[0], reverse=True)

    return [
        document
        for _, document in scored_documents[:limit]
    ]


def retrieve_travel_knowledge_from_vector_documents(
    query: str,
    vector_documents: list[VectorKnowledgeDocument],
    embed_query: EmbedTextFunc,
    city: str = "",
    limit: int = 3,
    min_score: float = 0.5,
) -> list[TravelKnowledgeDocument]:
    if not query.strip() or not vector_documents:
        return []

    query_vector = embed_query(query)
    scored_documents: list[tuple[float, TravelKnowledgeDocument]] = []

    for vector_document in vector_documents:
        document = vector_document.document
        score = cosine_similarity(query_vector, vector_document.vector)

        document_city = str(document.metadata.get("city", ""))
        if city and document_city == city:
            score += 0.05

        if score >= min_score:
            scored_documents.append((score, document))

    scored_documents.sort(key=lambda item: item[0], reverse=True)

    return [
        document
        for _, document in scored_documents[:limit]
    ]
