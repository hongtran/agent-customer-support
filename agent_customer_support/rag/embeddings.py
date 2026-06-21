from functools import lru_cache

from google import genai
from google.genai import types

from agent_customer_support.config import get_settings


@lru_cache
def _client() -> genai.Client:
    return genai.Client(api_key=get_settings().google_api_key)


async def embed_query(text: str) -> list[float]:
    """Embed a search query with the same model/params used to index the
    collection, so the query vector lands in the same space.

    task_type RETRIEVAL_QUERY mirrors how documents were indexed
    (RETRIEVAL_DOCUMENT); output_dimensionality must match the collection's
    vector size or the search mismatches.
    """
    cfg = get_settings()
    resp = await _client().aio.models.embed_content(
        model=cfg.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
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
