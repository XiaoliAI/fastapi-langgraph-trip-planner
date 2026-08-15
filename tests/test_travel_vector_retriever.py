from backend.app.rag.travel_knowledge_loader import TravelKnowledgeDocument
from backend.app.rag.vector_retriever import retrieve_travel_knowledge_by_vector


def fake_embed(text: str) -> list[float]:
    if "老人" in text or "轻松" in text or "少走路" in text or "减少步行" in text:
        return [1.0, 0.0, 0.0]

    if "美食" in text or "烤鸭" in text:
        return [0.0, 1.0, 0.0]

    return [0.0, 0.0, 1.0]


def test_vector_retriever_matches_semantic_document():
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
            content="北京美食建议：北京烤鸭、炸酱面适合安排进晚餐。",
            metadata={
                "source": "beijing_food.md",
                "city": "北京",
                "category": "food",
            },
        ),
    ]

    results = retrieve_travel_knowledge_by_vector(
        query="我带老人去北京，希望轻松一点，少走路",
        documents=documents,
        embed_query=fake_embed,
        embed_document=fake_embed,
        city="北京",
        limit=1,
    )

    assert len(results) == 1
    assert results[0].metadata["category"] == "low_walking"


def test_vector_retriever_returns_empty_for_unrelated_query():
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

    results = retrieve_travel_knowledge_by_vector(
        query="我想学习编程",
        documents=documents,
        embed_query=fake_embed,
        embed_document=fake_embed,
        city="北京",
        limit=1,
    )

    assert results == []
