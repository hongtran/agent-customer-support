from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent_customer_support.channels.deps import get_qa_indexer, get_qa_store, require_admin
from agent_customer_support.models import QARecord
from agent_customer_support.rag.qa_indexer import QAIndexer
from agent_customer_support.stores.qa_store import QAStore

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


class QACreate(BaseModel):
    question: str
    answer: str = ""
    application: str | None = None


class QAPatch(BaseModel):
    question: str | None = None
    answer: str | None = None
    application: str | None = None


class ApproveBody(BaseModel):
    approved_by: str | None = None


@router.get("/qa")
async def list_qa(status: str | None = None, qa: QAStore = Depends(get_qa_store)) -> list[QARecord]:
    return await qa.list(status=status)


@router.get("/qa/{record_id}")
async def get_qa(record_id: str, qa: QAStore = Depends(get_qa_store)) -> QARecord:
    rec = await qa.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    return rec


@router.post("/qa")
async def create_qa(body: QACreate, qa: QAStore = Depends(get_qa_store)) -> QARecord:
    return await qa.add(
        QARecord(
            question=body.question,
            answer=body.answer,
            application=body.application,
            source="manual",
        )
    )


@router.patch("/qa/{record_id}")
async def edit_qa(
    record_id: str,
    patch: QAPatch,
    qa: QAStore = Depends(get_qa_store),
    indexer: QAIndexer = Depends(get_qa_indexer),
) -> QARecord:
    rec = await qa.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    if patch.question is not None:
        rec.question = patch.question
    if patch.answer is not None:
        rec.answer = patch.answer
    if patch.application is not None:
        rec.application = patch.application
    await qa.update(rec)
    if rec.status == "approved":
        await indexer.upsert(rec)
    return rec


@router.post("/qa/{record_id}/approve")
async def approve_qa(
    record_id: str,
    body: ApproveBody,
    qa: QAStore = Depends(get_qa_store),
    indexer: QAIndexer = Depends(get_qa_indexer),
) -> QARecord:
    rec = await qa.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    if not rec.answer.strip():
        raise HTTPException(status_code=409, detail="answer required before approval")
    await indexer.upsert(rec)  # index-first: only persist approval if this succeeds
    rec.status = "approved"
    rec.approved_by = body.approved_by
    rec.indexed_at = datetime.now(UTC)
    rec.qdrant_point_id = rec.id
    return await qa.update(rec)


@router.post("/qa/{record_id}/reject")
async def reject_qa(record_id: str, qa: QAStore = Depends(get_qa_store)) -> QARecord:
    rec = await qa.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    rec.status = "rejected"
    return await qa.update(rec)


@router.post("/qa/{record_id}/archive")
async def archive_qa(
    record_id: str,
    qa: QAStore = Depends(get_qa_store),
    indexer: QAIndexer = Depends(get_qa_indexer),
) -> QARecord:
    rec = await qa.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    await indexer.delete(rec.id)
    rec.status = "archived"
    rec.qdrant_point_id = None
    return await qa.update(rec)
