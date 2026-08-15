import logging
from typing import Any

import httpx

from ..services.llm_service import _read_env_value


logger = logging.getLogger(__name__)

DEFAULT_JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
DEFAULT_JINA_EMBEDDING_MODEL = "jina-embeddings-v4"
DEFAULT_TIMEOUT_SECONDS = 30.0


def _get_http_client():
    return httpx


def _get_jina_api_key() -> str:
    api_key = _read_env_value("JINA_API_KEY") or _read_env_value("EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY or EMBEDDING_API_KEY is not configured")
    return api_key


def _get_jina_embedding_url() -> str:
    return (
        _read_env_value("JINA_EMBEDDING_BASE_URL")
        or _read_env_value("EMBEDDING_BASE_URL")
        or DEFAULT_JINA_EMBEDDING_URL
    )


def _get_jina_embedding_model() -> str:
    return (
        _read_env_value("JINA_EMBEDDING_MODEL")
        or _read_env_value("EMBEDDING_MODEL_ID")
        or DEFAULT_JINA_EMBEDDING_MODEL
    )


def _extract_embedding(response_data: dict[str, Any]) -> list[float]:
    data = response_data.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("Embedding response does not contain data")

    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("Embedding response item is not an object")

    embedding = first.get("embedding")
    if not isinstance(embedding, list):
        raise ValueError("Embedding response does not contain embedding")

    return [float(value) for value in embedding]


def _embed_with_task(text: str, task: str) -> list[float]:
    if not text.strip():
        return []

    url = _get_jina_embedding_url()
    model = _get_jina_embedding_model()
    api_key = _get_jina_api_key()

    payload = {
        "model": model,
        "task": task,
        "input": [
            {
                "text": text
            }
        ],
    }

    logger.info("Calling Jina embedding API: model=%s task=%s", model, task)

    response = _get_http_client().post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    vector = _extract_embedding(response.json())
    logger.info("Jina embedding returned vector: dimensions=%s task=%s", len(vector), task)

    return vector


def embed_query(text: str) -> list[float]:
    return _embed_with_task(text, task="retrieval.query")


def embed_document(text: str) -> list[float]:
    return _embed_with_task(text, task="retrieval.passage")