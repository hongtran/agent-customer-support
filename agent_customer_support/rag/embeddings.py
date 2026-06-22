from functools import lru_cache

from google import genai
from google.genai import types

from agent_customer_support.config import get_settings


@lru_cache
def _client() -> genai.Client:
    return genai.Client(api_key=get_settings().google_api_key)


async def _embed(text: str, task_type: str) -> list[float]:
    cfg = get_settings()
    resp = await _client().aio.models.embed_content(
        model=cfg.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=cfg.embedding_dim,
        ),
    )
    embeddings = resp.embeddings
    if not embeddings:
        raise ValueError("Embedding response contained no embeddings")
    values = embeddings[0].values
    if values is None:
        raise ValueError("Embedding response contained no values")
    return list(values)


async def embed_query(text: str) -> list[float]:
    """Embed a search query (RETRIEVAL_QUERY) — matches the indexed documents'
    space so search lands correctly."""
    return await _embed(text, "RETRIEVAL_QUERY")


async def embed_document(text: str) -> list[float]:
    """Embed a document/stored question (RETRIEVAL_DOCUMENT) for the qa index,
    paired with RETRIEVAL_QUERY at search time."""
    return await _embed(text, "RETRIEVAL_DOCUMENT")
