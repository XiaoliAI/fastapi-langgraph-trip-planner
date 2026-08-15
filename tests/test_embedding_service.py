from backend.app.rag import embedding_service
class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class FakeHttpClient:
    def __init__(self):
        self.requests = []

    def post(self, url, headers, json, timeout):
        self.requests.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse(
            {
                "data": [
                    {
                        "embedding": [0.1, 0.2, 0.3]
                    }
                ]
            }
        )


def test_embed_query_uses_jina_retrieval_query_task(monkeypatch):
    fake_client = FakeHttpClient()

    monkeypatch.setattr(
        embedding_service,
        "_get_http_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        embedding_service,
        "_read_env_value",
        lambda key: {
            "JINA_API_KEY": "test-key",
            "JINA_EMBEDDING_MODEL": "jina-embeddings-v4",
            "JINA_EMBEDDING_BASE_URL": "https://api.jina.ai/v1/embeddings",
        }.get(key),
    )

    vector = embedding_service.embed_query("我带老人去北京，希望少走路")

    assert vector == [0.1, 0.2, 0.3]

    request = fake_client.requests[0]
    assert request["url"] == "https://api.jina.ai/v1/embeddings"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["json"]["model"] == "jina-embeddings-v4"
    assert request["json"]["task"] == "retrieval.query"
    assert request["json"]["input"] == [
        {
            "text": "我带老人去北京，希望少走路"
        }
    ]


def test_embed_document_uses_jina_retrieval_passage_task(monkeypatch):
    fake_client = FakeHttpClient()

    monkeypatch.setattr(
        embedding_service,
        "_get_http_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        embedding_service,
        "_read_env_value",
        lambda key: {
            "JINA_API_KEY": "test-key",
            "JINA_EMBEDDING_MODEL": "jina-embeddings-v4",
            "JINA_EMBEDDING_BASE_URL": "https://api.jina.ai/v1/embeddings",
        }.get(key),
    )

    vector = embedding_service.embed_document(
        "北京老人旅行建议：减少步行，优先安排同一区域景点。"
    )

    assert vector == [0.1, 0.2, 0.3]

    request = fake_client.requests[0]
    assert request["json"]["task"] == "retrieval.passage"
    assert request["json"]["input"] == [
        {
            "text": "北京老人旅行建议：减少步行，优先安排同一区域景点。"
        }
    ]

