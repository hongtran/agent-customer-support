"""Parity check: compare the in-repo Qdrant read path against the legacy
external /rag/query for the same queries and collection.

Requires live env: QDRANT_ENDPOINT, QDRANT_API_KEY, GOOGLE_API_KEY, and
LEGACY_RAG_BASE_URL pointing at the still-running enterprise-llm-service.

Usage:
    poetry run python scripts/rag_parity_check.py "cách tạo mẫu xét nghiệm"
"""

import asyncio
import os
import sys

import httpx

from agent_customer_support.config import get_settings
from agent_customer_support.rag_client import RagClient

QUERIES = [
    "cách tạo mẫu xét nghiệm",
    "đổi mật khẩu",
    "khôi phục tài khoản",
    "Làm sao xử lý PYC sự cố?",
    "Làm thế nào để hủy một PYCTN?",
]


async def legacy(query: str, collection: str) -> dict:
    base = os.environ["LEGACY_RAG_BASE_URL"]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{base}/rag/query",
            json={
                "query": query,
                "collection_name": collection,
                "top_k": 8,
                "score_threshold": 0.6,
                "doc_type": None,
                "applications": [],
            },
        )
        resp.raise_for_status()
        data = resp.json()
    metas = data.get("metadatas", []) or []
    cites = sorted(
        {m.get("source_doc_id") or m.get("doc_id", "") for m in metas if m.get("source_doc_id") or m.get("doc_id")}
    )
    confs = {m.get("source_doc_id") or m.get("doc_id", ""): m.get("confidence", 0.0) for m in metas}
    return {"citations": cites, "confs": confs}


async def main() -> None:
    queries = sys.argv[1:] or QUERIES
    collection = get_settings().product_collection
    rag = RagClient()
    mismatches = 0
    for q in queries:
        new = await rag.search(q, collection=collection)
        old = await legacy(q, collection)
        new_cites = set(new["citations"])
        old_cites = set(old["citations"])
        same = new_cites == old_cites
        mismatches += 0 if same else 1
        print(f"\nQ: {q}")
        print(f"  new citations: {sorted(new_cites)}")
        print(f"  old citations: {sorted(old_cites)}")
        print(f"  match: {same}")
        if not same:
            print(f"  only new: {new_cites - old_cites}")
            print(f"  only old: {old_cites - new_cites}")
    print(f"\n{len(queries) - mismatches}/{len(queries)} queries matched.")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
