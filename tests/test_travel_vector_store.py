from backend.app.rag.travel_knowledge_loader import TravelKnowledgeDocument
from backend.app.rag.vector_store import (
    clear_vector_knowledge_cache,
    get_vector_knowledge_documents,
)


def test_get_vector_knowledge_documents_caches_document_vectors():
    clear_vector_knowledge_cache()

    documents = [
        TravelKnowledgeDocument(
            content="北京老人旅行建议：减少步行。",
            metadata={
                "source": "beijing_senior.md",
                "city": "北京",
                "category": "low_walking",
            },
        )
    ]
    calls = []

    def fake_embed_document(text: str) -> list[float]:
        calls.append(text)
        return [1.0, 0.0, 0.0]

    first_result = get_vector_knowledge_documents(
        documents=documents,
        embed_document=fake_embed_document,
    )
    second_result = get_vector_knowledge_documents(
        documents=documents,
        embed_document=fake_embed_document,
    )

    assert len(calls) == 1
    assert first_result[0].vector == [1.0, 0.0, 0.0]
    assert second_result[0].vector == [1.0, 0.0, 0.0]


def test_get_vector_knowledge_documents_reembeds_changed_document_content():
    clear_vector_knowledge_cache()

    calls = []

    def fake_embed_document(text: str) -> list[float]:
        calls.append(text)
        return [float(len(calls)), 0.0, 0.0]

    first_documents = [
        TravelKnowledgeDocument(
            content="第一版内容",
            metadata={"source": "guide.md", "city": "北京", "category": "tips"},
        )
    ]
    second_documents = [
        TravelKnowledgeDocument(
            content="第二版内容",
            metadata={"source": "guide.md", "city": "北京", "category": "tips"},
        )
    ]

    first_result = get_vector_knowledge_documents(
        documents=first_documents,
        embed_document=fake_embed_document,
    )
    second_result = get_vector_knowledge_documents(
        documents=second_documents,
        embed_document=fake_embed_document,
    )

    assert len(calls) == 2
    assert first_result[0].vector == [1.0, 0.0, 0.0]
    assert second_result[0].vector == [2.0, 0.0, 0.0]
