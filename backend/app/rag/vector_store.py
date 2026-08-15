#这段代码实现了一个带缓存功能的文档向量化加载器

from dataclasses import dataclass
from typing import Callable

from .travel_knowledge_loader import TravelKnowledgeDocument


EmbedTextFunc = Callable[[str], list[float]]


@dataclass(frozen=True)
class VectorKnowledgeDocument:
    document: TravelKnowledgeDocument
    vector: list[float]


_document_vector_cache: dict[str, VectorKnowledgeDocument] = {}


def build_document_cache_key(document: TravelKnowledgeDocument) -> str:
    source = document.metadata.get("source", "")
    city = document.metadata.get("city", "")
    category = document.metadata.get("category", "")
    return f"{source}|{city}|{category}|{document.content}"


def get_vector_knowledge_documents(
    documents: list[TravelKnowledgeDocument],
    embed_document: EmbedTextFunc,
) -> list[VectorKnowledgeDocument]:
    vector_documents: list[VectorKnowledgeDocument] = []

    for document in documents:
        cache_key = build_document_cache_key(document)

        if cache_key not in _document_vector_cache:
            _document_vector_cache[cache_key] = VectorKnowledgeDocument(
                document=document,
                vector=embed_document(document.content),
            )

        vector_documents.append(_document_vector_cache[cache_key])

    return vector_documents


def clear_vector_knowledge_cache():
    _document_vector_cache.clear()
