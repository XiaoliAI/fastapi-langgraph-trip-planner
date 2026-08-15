from pathlib import Path

from backend.app.rag.travel_knowledge_loader import (
    TravelKnowledgeDocument,
    infer_city_from_file_name,
    load_travel_knowledge_documents,
    load_default_travel_knowledge_documents,
)


def test_infer_city_from_file_name():
    assert infer_city_from_file_name("beijing_family.md") == "北京"
    assert infer_city_from_file_name("beijing_food.md") == "北京"
    assert infer_city_from_file_name("general_budget_tips.md") == "general"


def test_load_travel_knowledge_documents_reads_markdown_files(tmp_path):
    knowledge_dir = tmp_path / "travel_knowledge"
    knowledge_dir.mkdir()

    (knowledge_dir / "beijing_family.md").write_text(
        "# 北京亲子旅行建议\n\n适合亲子旅行。",
        encoding="utf-8",
    )
    (knowledge_dir / "general_low_walking_tips.md").write_text(
        "# 少走路旅行建议\n\n减少跨区域移动。",
        encoding="utf-8",
    )

    documents = load_travel_knowledge_documents(knowledge_dir)

    assert len(documents) == 2
    assert all(isinstance(document, TravelKnowledgeDocument) for document in documents)

    sources = {document.metadata["source"] for document in documents}
    assert sources == {
        "beijing_family.md",
        "general_low_walking_tips.md",
    }

    beijing_doc = next(
        document
        for document in documents
        if document.metadata["source"] == "beijing_family.md"
    )

    assert beijing_doc.metadata["city"] == "北京"
    assert beijing_doc.metadata["category"] == "family"
    assert "亲子旅行" in beijing_doc.content


def test_load_travel_knowledge_documents_returns_empty_for_missing_dir(tmp_path):
    missing_dir = tmp_path / "missing"

    documents = load_travel_knowledge_documents(missing_dir)

    assert documents == []
def test_load_default_travel_knowledge_documents_reads_project_knowledge_files():
    documents = load_default_travel_knowledge_documents()

    assert len(documents) >= 4

    sources = {document.metadata["source"] for document in documents}

    assert "beijing_family.md" in sources
    assert "beijing_food.md" in sources
    assert "general_budget_tips.md" in sources
    assert "general_low_walking_tips.md" in sources