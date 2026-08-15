from backend.app.rag.travel_knowledge_loader import TravelKnowledgeDocument
from backend.app.rag.travel_retriever import (
    retrieve_travel_knowledge,
    format_retrieved_knowledge,
    retrieve_default_travel_knowledge_text,
    
)


def make_documents():
    return [
        TravelKnowledgeDocument(
            content="少走路旅行建议：减少跨区域移动，优先安排同一区域景点。",
            metadata={
                "source": "general_low_walking_tips.md",
                "city": "general",
                "category": "low_walking_tips",
            },
        ),
        TravelKnowledgeDocument(
            content="北京美食旅行建议：北京烤鸭、炸酱面、涮羊肉适合安排进晚餐。",
            metadata={
                "source": "beijing_food.md",
                "city": "北京",
                "category": "food",
            },
        ),
        TravelKnowledgeDocument(
            content="预算控制旅行建议：优先安排免费或低门票景点。",
            metadata={
                "source": "general_budget_tips.md",
                "city": "general",
                "category": "budget_tips",
            },
        ),
    ]


def test_retrieve_travel_knowledge_matches_query_text():
    documents = make_documents()

    results = retrieve_travel_knowledge(
        query="我想少走路，轻松一点",
        documents=documents,
        city="北京",
        limit=2,
    )

    assert len(results) >= 1
    assert results[0].metadata["source"] == "general_low_walking_tips.md"


def test_retrieve_travel_knowledge_prioritizes_city_specific_documents():
    documents = make_documents()

    results = retrieve_travel_knowledge(
        query="想多吃本地美食",
        documents=documents,
        city="北京",
        limit=2,
    )

    assert len(results) >= 1
    assert results[0].metadata["source"] == "beijing_food.md"


def test_retrieve_travel_knowledge_returns_empty_for_unrelated_query():
    documents = make_documents()

    results = retrieve_travel_knowledge(
        query="我要学习编程",
        documents=documents,
        city="北京",
        limit=2,
    )

    assert results == []


def test_format_retrieved_knowledge():
    documents = make_documents()

    text = format_retrieved_knowledge(documents[:2])

    assert "general_low_walking_tips.md" in text
    assert "beijing_food.md" in text
    assert "少走路旅行建议" in text
    assert "北京美食旅行建议" in text

def test_retrieve_default_travel_knowledge_text_reads_project_documents():
    text = retrieve_default_travel_knowledge_text(
        query="我想少走路，轻松一点",
        city="北京",
        limit=2,
    )

    assert "general_low_walking_tips.md" in text
    assert "少走路旅行建议" in text

def test_retrieve_default_travel_knowledge_by_vector_text(monkeypatch):
    from backend.app.rag import travel_retriever

    documents = [
        TravelKnowledgeDocument(
            content="北京老人旅行建议：减少步行，优先安排同一区域景点。",
            metadata={
                "source": "beijing_senior.md",
                "city": "北京",
                "category": "low_walking",
            },
        ),
        TravelKnowledgeDocument(
            content="北京美食建议：北京烤鸭适合安排进晚餐。",
            metadata={
                "source": "beijing_food.md",
                "city": "北京",
                "category": "food",
            },
        ),
    ]

    def fake_embed_query(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def fake_embed_document(text: str) -> list[float]:
        if "老人" in text or "减少步行" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr(
        travel_retriever,
        "load_default_travel_knowledge_documents",
        lambda: documents,
    )
    monkeypatch.setattr(
        travel_retriever,
        "embed_query",
        fake_embed_query,
    )
    monkeypatch.setattr(
        travel_retriever,
        "embed_document",
        fake_embed_document,
    )

    text = travel_retriever.retrieve_default_travel_knowledge_by_vector_text(
        query="我带老人去北京，希望轻松一点",
        city="北京",
        limit=1,
    )

    assert "beijing_senior.md" in text
    assert "减少步行" in text
    assert "beijing_food.md" not in text


def test_smart_retriever_falls_back_to_keyword_retriever(monkeypatch):
    from backend.app.rag import travel_retriever

    documents = make_documents()

    def fake_vector_retriever(**kwargs):
        raise RuntimeError("Jina API unavailable")

    monkeypatch.setattr(
        travel_retriever,
        "load_default_travel_knowledge_documents",
        lambda: documents,
    )
    monkeypatch.setattr(
        travel_retriever,
        "retrieve_travel_knowledge_by_vector",
        fake_vector_retriever,
    )

    text = travel_retriever.retrieve_default_travel_knowledge_smart_text(
        query="我想少走路，轻松一点",
        city="北京",
        limit=2,
    )

    assert "general_low_walking_tips.md" in text
    assert "少走路旅行建议" in text


def test_smart_retriever_reuses_cached_document_vectors(monkeypatch):
    from backend.app.rag import travel_retriever
    from backend.app.rag.vector_store import clear_vector_knowledge_cache

    clear_vector_knowledge_cache()
    documents = [
        TravelKnowledgeDocument(
            content="北京老人旅行建议：减少步行，优先安排同一区域景点。",
            metadata={
                "source": "beijing_senior.md",
                "city": "北京",
                "category": "low_walking",
            },
        )
    ]
    document_embedding_calls = []
    query_embedding_calls = []

    def fake_embed_document(text: str) -> list[float]:
        document_embedding_calls.append(text)
        return [1.0, 0.0, 0.0]

    def fake_embed_query(text: str) -> list[float]:
        query_embedding_calls.append(text)
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        travel_retriever,
        "load_default_travel_knowledge_documents",
        lambda: documents,
    )
    monkeypatch.setattr(
        travel_retriever,
        "embed_document",
        fake_embed_document,
    )
    monkeypatch.setattr(
        travel_retriever,
        "embed_query",
        fake_embed_query,
    )

    first_text = travel_retriever.retrieve_default_travel_knowledge_smart_text(
        query="我带老人去北京，希望轻松一点",
        city="北京",
        limit=1,
    )
    second_text = travel_retriever.retrieve_default_travel_knowledge_smart_text(
        query="希望少走路",
        city="北京",
        limit=1,
    )

    assert "beijing_senior.md" in first_text
    assert "beijing_senior.md" in second_text
    assert len(document_embedding_calls) == 1
    assert len(query_embedding_calls) == 2
