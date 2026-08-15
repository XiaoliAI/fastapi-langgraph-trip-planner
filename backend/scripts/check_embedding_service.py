from app.rag.embedding_service import embed_query, embed_document


def preview_vector(vector: list[float]) -> str:
    if not vector:
        return "EMPTY"

    first_values = ", ".join(f"{value:.4f}" for value in vector[:5])
    return f"dimensions={len(vector)}, first_values=[{first_values}]"


def main():
    query = "我带老人去北京，希望每天轻松一点，不要走太多路"
    document = "北京老人旅行建议：减少步行，优先安排同一区域景点，减少跨区域移动。"

    print("Calling embed_query...")
    query_vector = embed_query(query)
    print("Query vector:")
    print(preview_vector(query_vector))

    print("\nCalling embed_document...")
    document_vector = embed_document(document)
    print("Document vector:")
    print(preview_vector(document_vector))


if __name__ == "__main__":
    main()