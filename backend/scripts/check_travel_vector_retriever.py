from backend.app.rag.travel_knowledge_loader import load_default_travel_knowledge_documents
from backend.app.rag.travel_retriever import (
    retrieve_default_travel_knowledge_smart_text,
    retrieve_default_travel_knowledge_by_vector_text,
)


def main():
    query = "我带老人去北京，希望每天轻松一点，不要走太多路"
    city = "北京"

    documents = load_default_travel_knowledge_documents()
    print(f"Loaded documents: {len(documents)}")
    print(f"Query: {query}")
    print(f"City: {city}")

    print("\nSmart RAG result:")
    smart_text = retrieve_default_travel_knowledge_smart_text(
        query=query,
        city=city,
        limit=3,
    )
    print(smart_text)

    print("\nVector-only RAG result:")
    vector_text = retrieve_default_travel_knowledge_by_vector_text(
        query=query,
        city=city,
        limit=3,
    )
    print(vector_text)


if __name__ == "__main__":
    main()
