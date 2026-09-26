import base64

import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_customer_support.agents.coordinator import Coordinator
from tests.agents.composed import composed_answer
from agent_customer_support.agents.prompts import ASK_CONTACT_REPLY, CONTACT_THANKS_REPLY
from agent_customer_support.mantis import MantisIssue
from agent_customer_support.models import (
    AgentResult,
    BugSlots,
    Attachment,
    AttachmentRef,
    ContactInfo,
    CustomerProfile,
    RequestRecord,
    SessionState,
    Conversation,
    StoredAttachment,
    Turn,
)

pytestmark = pytest.mark.asyncio


def _stored(key: str, media_type: str = "image/png") -> StoredAttachment:
    return StoredAttachment(kind="image", media_type=media_type, s3_key=key, size_bytes=3)


def _coord():
    c = Coordinator()
    c.customers = AsyncMock()
    c.customers.get.return_value = CustomerProfile(
        customer_id="c1", name="C1", enabled_applications=["Lấy mẫu - Quan trắc"]
    )
    c.conversations = AsyncMock()
    c.conversations.load.return_value = Conversation(conversation_id="cv1", customer_id="c1")
    c.sessions = AsyncMock()
    c.sessions.get.return_value = SessionState(conversation_id="cv1")
    c.rag = AsyncMock()
    c.flow_store = AsyncMock()
    c.backlog = AsyncMock()
    c.escalator = AsyncMock()
    c.attachments = AsyncMock()
    c.attachments.put.return_value = _stored("cur")
    c.attachments.presign.return_value = AttachmentRef(
        kind="image", media_type="image/png", url="https://s3/cur"
    )
    c.mantis = AsyncMock()
    c.mantis.create_issue.return_value = None
    c.backlog.add.return_value = RequestRecord(id="r1", customer_id="c1", type="bug", summary="s")
    # agent stubs
    c.guardrail = MagicMock()
    c.guardrail.check_input = AsyncMock(return_value={"pass": True, "reason": ""})
    c.guardrail.check_output = AsyncMock(return_value={"pass": True, "reason": ""})
    c.triage = MagicMock()
    c.knowledge = MagicMock()
    c.issue_verification = MagicMock()
    c.escalation = MagicMock()
    return c


async def test_input_guardrail_block_short_circuits():
    c = _coord()
    c.guardrail.check_input = AsyncMock(return_value={"pass": False, "reason": "empty"})
    c.triage.run = AsyncMock()
    res = await c.handle_turn(customer_id="c1", conversation_id="cv1", message="  ", attachments=[])
    assert res.escalated is False
    c.triage.run.assert_not_called()


async def test_knowledge_clarify_resume_bypasses_triage():
    """When a knowledge clarification is pending, the next turn goes straight to
    knowledge (the answer to our clarify question), not back through triage."""
    c = _coord()
    c.sessions.get.return_value = SessionState(conversation_id="cv1", pending="knowledge_clarify")
    c.triage.run = AsyncMock()
    c.knowledge.run = AsyncMock(
        return_value=AgentResult(reply="Vào menu X.", knowledge_status="answer")
    )
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="trang xuất kho", attachments=[]
    )
    assert res.reply == "Vào menu X."
    c.triage.run.assert_not_called()
    c.knowledge.run.assert_awaited_once()


async def test_knowledge_clarify_roundtrip_no_escalate_then_resolves():
    """End-to-end clarify round-trip through the real KnowledgeAgent:

    Turn 1: compose reports status="clarify". The turn must NOT escalate, and must
            persist session.pending == "knowledge_clarify" with clarify_count spent.
    Turn 2: compose returns a plain grounded answer → the reply is the answer and
            the pending flag is cleared.
    """
    from agent_customer_support.agents.knowledge import KnowledgeAgent
    from agent_customer_support.agents import knowledge as knowledge_mod

    c = _coord()
    # Use the real KnowledgeAgent so the clarify wiring (the status it reports, and
    # the flag and counter the driver moves) is exercised, not stubbed away.
    c.knowledge = KnowledgeAgent()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.rag.search = AsyncMock(return_value={"passages": [], "citations": []})
    c.rag.search_with_fallback = c.rag.search  # product search entry point

    # A single mutable session that survives across both turns, mirroring how the
    # real SessionStore would carry pending state turn-to-turn.
    session = SessionState(conversation_id="cv1")
    c.sessions.get.return_value = session

    # Turn 1: compose emits a clarify marker.
    compose_outputs = iter(
        [
            "Bạn muốn tạo loại phiếu nào? [[clarify]]",
            "Vào menu Phiếu > Tạo mới.",
        ]
    )

    def fake_compose(*args, **kwargs):
        return composed_answer(next(compose_outputs))

    original_complete_structured = knowledge_mod.complete_structured
    knowledge_mod.complete_structured = fake_compose
    try:
        res1 = await c.handle_turn(
            customer_id="c1", conversation_id="cv1", message="tạo phiếu", attachments=[]
        )
        assert res1.escalated is False
        assert "Bạn muốn tạo loại phiếu nào?" in res1.reply
        saved1 = c.sessions.save.call_args.args[0]
        assert saved1.pending == "knowledge_clarify"
        assert saved1.clarify_count == 1

        # Turn 2: same session, pending flag set → coordinator resumes in knowledge.
        c.sessions.get.return_value = saved1
        res2 = await c.handle_turn(
            customer_id="c1", conversation_id="cv1", message="phiếu kết quả", attachments=[]
        )
        assert res2.escalated is False
        assert res2.reply == "Vào menu Phiếu > Tạo mới."
        saved2 = c.sessions.save.call_args.args[0]
        assert saved2.pending is None
    finally:
        knowledge_mod.complete_structured = original_complete_structured


async def test_root_trace_uses_conversation_as_session(monkeypatch):
    from contextlib import contextmanager
    from agent_customer_support.agents import coordinator as coord_mod

    captured = {}

    @contextmanager
    def fake_trace(name, *, session_id=None, user_id=None, tags=None, input=None, metadata=None):
        captured["session_id"] = session_id
        captured["user_id"] = user_id
        yield MagicMock()

    monkeypatch.setattr(coord_mod.tracing, "trace", fake_trace)
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="ok", knowledge_status="answer"))
    await c.handle_turn(customer_id="c1", conversation_id="cv1", message="?", attachments=[])
    assert captured["session_id"] == "cv1"
    assert captured["user_id"] == "c1"


async def test_knowledge_resolved_returns_reply():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="đáp án", knowledge_status="answer"))
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="cách làm X", attachments=[]
    )
    assert res.reply == "đáp án"


async def test_knowledge_unresolved_escalates():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="", knowledge_status="no_answer"))
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="chuyển nhân viên", escalated=True))
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="lỗi lạ", attachments=[]
    )
    assert res.escalated is True
    c.escalation.run.assert_awaited_once()


async def test_suspected_bug_starts_verification():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(
        return_value=AgentResult(
            reply="nghi lỗi",
            knowledge_status="suspected_bug",
            evidence={"application": "Lấy mẫu - Quan trắc", "summary": "A lỗi"},
        )
    )
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(reply="gửi ảnh giúp mình", verify_outcome="need_more_info")
    )
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="A bị lỗi", attachments=[]
    )
    assert "ảnh" in res.reply
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "verify_issue"


async def test_pending_verification_resumes_and_escalates_when_complete():
    c = _coord()
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1",
        pending="verify_issue",
        pending_context={"application": "Lấy mẫu - Quan trắc", "summary": "A lỗi"},
    )
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(
            reply="đã đủ",
            verify_outcome="bug_confirmed",
            evidence={"application": "Lấy mẫu - Quan trắc", "summary": "A lỗi", "has_image": True},
        )
    )
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="chuyển nhân viên", escalated=True))
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="đây là ảnh", attachments=[]
    )
    assert res.escalated is True
    c.backlog.add.assert_awaited_once()
    saved = c.sessions.save.call_args.args[0]
    # verification is over; what remains pending is the one-shot contact ask
    assert saved.pending == "collect_contact"
    assert "since_turn" not in (saved.pending_context or {})


async def test_out_of_scope_route_refuses_before_knowledge():
    """Triage flags off-topic: canonical refusal, no RAG/knowledge spend, no escalation."""
    from agent_customer_support.agents.prompts import OUT_OF_SCOPE_REPLY

    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="out_of_scope"))
    c.knowledge.run = AsyncMock()
    c.escalation.run = AsyncMock()
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="Tỷ giá USD hôm nay?", attachments=[]
    )
    assert res.reply == OUT_OF_SCOPE_REPLY
    assert res.escalated is False
    c.knowledge.run.assert_not_called()
    c.escalation.run.assert_not_called()


# ---- output guardrail: repair ladder before escalating ----

from agent_customer_support.models import Citation  # noqa: E402

_CITED = [Citation(doc_id="d1", label="Tạo phiếu", kind="guide")]
_ANSWER = "Đang sử dụng, Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu."


def _flagged_coord(claims: list[dict]):
    c = _coord()
    c.sessions.get.return_value = SessionState(conversation_id="cv1", pending="knowledge_clarify")
    c.knowledge.run = AsyncMock(
        return_value=AgentResult(
            reply=_ANSWER,
            knowledge_status="answer",
            citations=_CITED,
            source_passages=["Vào menu Phiếu yêu cầu."],
        )
    )
    c.knowledge.repair = AsyncMock(return_value=None)
    c.guardrail.check_output = AsyncMock(
        return_value={"pass": False, "reason": "x", "unsupported_claims": claims}
    )
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="Đã chuyển CS.", escalated=True))
    return c


async def _turn(c):
    return await c.handle_turn(customer_id="c1", conversation_id="cv1", message="q", attachments=[])


async def test_minor_claims_are_deleted_in_python_without_a_second_judge_call():
    c = _flagged_coord([{"span": "Đang sử dụng, ", "severity": "minor", "reason": "thừa"}])
    res = await _turn(c)
    assert res.reply == "Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu."
    assert res.escalated is False
    # The sources still vouch for what remains, so they stay attached.
    assert res.citations == _CITED
    assert c.guardrail.check_output.await_count == 1
    c.knowledge.repair.assert_not_awaited()
    c.escalation.run.assert_not_awaited()


async def test_a_span_python_cannot_delete_goes_to_the_llm_repair_and_is_judged_again():
    # Ends with a full stop: a whole sentence, which apply_claims refuses.
    c = _flagged_coord([{"span": "để lập phiếu.", "severity": "minor", "reason": "thừa"}])
    c.knowledge.repair = AsyncMock(return_value="Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới.")
    c.guardrail.check_output = AsyncMock(
        side_effect=[
            {
                "pass": False,
                "reason": "x",
                "unsupported_claims": [
                    {"span": "để lập phiếu.", "severity": "minor", "reason": "thừa"}
                ],
            },
            {"pass": True, "reason": ""},
        ]
    )
    res = await _turn(c)
    assert res.reply == "Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới."
    assert res.escalated is False
    assert res.citations == _CITED
    # Repaired against the same sources the original answer cited.
    c.knowledge.repair.assert_awaited_once_with(
        _ANSWER,
        [{"span": "để lập phiếu.", "severity": "minor", "reason": "thừa"}],
        ["Vào menu Phiếu yêu cầu."],
    )
    second = c.guardrail.check_output.await_args_list[1]
    assert second.args == (
        "Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới.",
        ["Vào menu Phiếu yêu cầu."],
    )
    # Both judge calls see the customer's question, the first and the recheck alike.
    assert [call.kwargs["question"] for call in c.guardrail.check_output.await_args_list] == [
        "q",
        "q",
    ]
    c.escalation.run.assert_not_awaited()


async def test_a_repair_that_still_fails_escalates():
    c = _flagged_coord([{"span": "để lập phiếu.", "severity": "minor", "reason": "thừa"}])
    c.knowledge.repair = AsyncMock(return_value="vẫn sai")
    res = await _turn(c)
    assert res.escalated is True
    assert res.citations == []
    assert "vẫn sai" not in res.reply
    assert c.guardrail.check_output.await_count == 2


async def test_a_repair_that_returns_nothing_escalates():
    c = _flagged_coord([{"span": "để lập phiếu.", "severity": "minor", "reason": "thừa"}])
    res = await _turn(c)
    assert res.escalated is True
    assert c.guardrail.check_output.await_count == 1


async def test_a_major_claim_skips_python_and_goes_to_the_llm_repair():
    c = _flagged_coord(
        [
            {"span": "Đang sử dụng, ", "severity": "minor", "reason": "thừa"},
            {"span": "Tạo mới", "severity": "major", "reason": "sai nút"},
        ]
    )
    res = await _turn(c)
    # The repair mock returns None, so the turn still hands off -- but only after trying.
    c.knowledge.repair.assert_awaited_once()
    assert res.escalated is True
    assert res.citations == []
    assert c.guardrail.check_output.await_count == 1


async def test_a_failure_with_no_named_claims_escalates():
    c = _flagged_coord([])
    res = await _turn(c)
    assert res.escalated is True
    c.knowledge.repair.assert_not_awaited()


async def test_a_minor_claim_with_a_replacement_is_applied_in_python():
    c = _flagged_coord(
        [
            {
                "span": "Đang sử dụng, Anh/Chị",
                "replacement": "Anh/Chị",
                "severity": "minor",
                "reason": "thừa",
            }
        ]
    )
    res = await _turn(c)
    assert res.reply == "Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu."
    assert res.escalated is False
    c.knowledge.repair.assert_not_awaited()


_REPORT = {
    "title": "Import ký hiệu mẫu báo lỗi 500",
    "summary": "Import file mẫu thì báo lỗi 500.",
    "steps_to_reproduce": "1. Vào Lấy mẫu\n2. Bấm Import",
}


def _verified_bug_coord(*, since_turn: int = 2):
    """A conversation where the bug was suspected at turn index 2: a screenshot before
    it (turn 0) is shown to the verifier but must NOT reach the ticket; the one sent
    during verification (turn 4) must."""
    c = _coord()
    c.conversations.load.return_value = Conversation(
        conversation_id="cv1",
        customer_id="c1",
        turns=[
            Turn(role="user", content="hỏi cũ", attachments=[_stored("old.png")]),
            Turn(role="assistant", content="đáp cũ"),
            Turn(role="user", content="A lỗi"),
            Turn(role="assistant", content="gửi ảnh giúp mình"),
            Turn(role="user", content="đây ạ", attachments=[_stored("k1.jpg", "image/jpeg")]),
            Turn(role="assistant", content="cần thêm thông báo lỗi"),
        ],
    )
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1",
        pending="verify_issue",
        pending_context={
            "application": "Lấy mẫu - Quan trắc",
            "summary": "A lỗi",
            "since_turn": since_turn,
        },
    )
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(
            reply="đã đủ",
            verify_outcome="bug_confirmed",
            evidence={
                "application": "Lấy mẫu - Quan trắc",
                "summary": "A lỗi",
                "since_turn": since_turn,
                "has_image": True,
                "report": _REPORT,
            },
        )
    )
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="chuyển nhân viên", escalated=True))
    c.attachments.get_bytes = AsyncMock(return_value=b"k1-bytes")
    return c


async def test_verified_bug_files_ticket_with_evidence_then_logs_and_escalates():
    c = _verified_bug_coord()
    c.mantis.create_issue.return_value = MantisIssue(
        id=7, url="https://mantis.example/view.php?id=7"
    )
    cur = Attachment(kind="image", media_type="image/png", data="QUJD")
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="lỗi 500", attachments=[cur]
    )
    assert res.escalated is True

    kw = c.mantis.create_issue.call_args.kwargs
    assert kw["report"].title == _REPORT["title"]
    assert kw["customer_id"] == "c1"
    assert kw["customer_name"] == "C1"
    assert kw["application"] == "Lấy mẫu - Quan trắc"
    assert kw["transcript"].endswith("user: lỗi 500")
    # prior screenshot from verification (read back from S3) first, current turn last;
    # the unrelated one from before the bug was suspected is absent
    assert [(f.name, f.content_b64) for f in kw["files"]] == [
        ("screenshot-1.jpg", "azEtYnl0ZXM="),
        ("screenshot-2.png", "QUJD"),
    ]
    # The verifier reads from the start of the conversation (old.png included); the
    # ticket reads only from since_turn, so old.png is read once and never filed.
    keys = [c.args[0].s3_key for c in c.attachments.get_bytes.await_args_list]
    assert sorted(keys) == ["k1.jpg", "k1.jpg", "old.png"]

    bk = c.backlog.add.call_args.kwargs
    assert bk["type"] == "bug"
    assert bk["title"] == _REPORT["title"]
    assert bk["summary"].startswith(_REPORT["summary"])
    assert bk["mantis_issue_id"] == 7
    assert bk["mantis_issue_url"] == "https://mantis.example/view.php?id=7"

    ek = c.escalation.run.call_args.kwargs
    assert ek["reason"] == "verified bug"
    assert "https://mantis.example/view.php?id=7" in ek["note"]

    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "collect_contact"
    assert "since_turn" not in (saved.pending_context or {})


async def test_verified_bug_without_ticket_still_logs_and_escalates():
    c = _verified_bug_coord()
    c.mantis.create_issue.return_value = None
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="lỗi 500", attachments=[]
    )
    assert res.escalated is True
    bk = c.backlog.add.call_args.kwargs
    assert bk["mantis_issue_id"] is None and bk["mantis_issue_url"] is None
    assert bk["title"] == _REPORT["title"]
    note = c.escalation.run.call_args.kwargs["note"]
    assert "KHÔNG tạo được" in note


async def test_verified_bug_without_report_uses_fallback_title():
    """Evidence written before the report step existed (or a session that predates
    the deploy) has no `report`; the ticket is still filed from the raw summary."""
    c = _verified_bug_coord()
    c.issue_verification.run.return_value.evidence.pop("report")
    await c.handle_turn(customer_id="c1", conversation_id="cv1", message="lỗi 500", attachments=[])
    kw = c.mantis.create_issue.call_args.kwargs
    assert kw["report"].title == "A lỗi" and kw["report"].summary.startswith("A lỗi")
    assert c.backlog.add.call_args.kwargs["title"] == "A lỗi"


async def test_the_ticket_body_names_what_was_collected_and_what_is_missing():
    """An engineer reading the ticket has to be able to tell a thin report from a
    complete one, especially when the collection was cut short by the turn cap."""
    c = _verified_bug_coord()
    c.issue_verification.run.return_value.evidence["slots"] = (
        BugSlots.empty()
        .model_copy(update={"steps": "1. Bấm Import", "actual": "Báo lỗi 500"})
        .model_dump()
    )
    await c.handle_turn(customer_id="c1", conversation_id="cv1", message="lỗi 500", attachments=[])
    summary = c.mantis.create_issue.call_args.kwargs["report"].summary
    assert "1. Bấm Import" in summary
    assert "ảnh chụp màn hình: có" in summary
    # `module` was never filled, and the ticket says so rather than reading complete.
    assert "Thiếu thông tin: màn hình/menu" in summary


async def test_evidence_read_failure_drops_that_file_only():
    c = _verified_bug_coord()
    c.attachments.get_bytes = AsyncMock(side_effect=RuntimeError("s3 down"))
    cur = Attachment(kind="image", media_type="image/png", data="QUJD")
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="lỗi 500", attachments=[cur]
    )
    assert res.escalated is True
    files = c.mantis.create_issue.call_args.kwargs["files"]
    assert [(f.name, f.content_b64) for f in files] == [("screenshot-1.png", "QUJD")]


async def test_evidence_files_are_capped_to_the_most_recent():
    c = _verified_bug_coord(since_turn=0)
    c.conversations.load.return_value = Conversation(
        conversation_id="cv1",
        customer_id="c1",
        turns=[
            Turn(role="user", content=f"ảnh {i}", attachments=[_stored(f"k{i}.png")])
            for i in range(7)
        ],
    )
    c.attachments.get_bytes = AsyncMock(side_effect=lambda st: st.s3_key.encode())
    cur = Attachment(kind="image", media_type="image/png", data="QUJD")
    await c.handle_turn(customer_id="c1", conversation_id="cv1", message="lỗi", attachments=[cur])
    files = c.mantis.create_issue.call_args.kwargs["files"]
    assert len(files) == 5  # settings.mantis_max_files default
    # most recent kept: k3..k6 from history, then the current turn
    assert files[-1].content_b64 == "QUJD"
    assert files[0].content_b64 == base64.b64encode(b"k3.png").decode()


async def test_suspected_bug_records_where_verification_started():
    c = _coord()
    c.conversations.load.return_value = Conversation(
        conversation_id="cv1",
        customer_id="c1",
        turns=[Turn(role="user", content="cũ"), Turn(role="assistant", content="đáp")],
    )
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(
        return_value=AgentResult(
            reply="nghi lỗi",
            knowledge_status="suspected_bug",
            evidence={"application": "Lấy mẫu - Quan trắc", "summary": "A lỗi"},
        )
    )
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(reply="gửi ảnh giúp mình", verify_outcome="need_more_info")
    )
    await c.handle_turn(customer_id="c1", conversation_id="cv1", message="A bị lỗi", attachments=[])
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "verify_issue"
    # index the current (not yet persisted) user turn will get
    assert saved.pending_context["since_turn"] == 2
    assert saved.pending_context["application"] == "Lấy mẫu - Quan trắc"


# ---- contact ask after a handoff --------------------------------------------------


def _escalation_result(reason: str) -> AgentResult:
    return AgentResult(reply="chuyển nhân viên", escalated=True, escalation_reason=reason)


async def _human_requested_turn(c):
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="escalate"))
    c.escalation.run = AsyncMock(return_value=_escalation_result("user requested human"))
    return await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="cho gặp nhân viên", attachments=[]
    )


async def test_user_requested_human_hands_off_then_asks_for_contact():
    c = _coord()
    res = await _human_requested_turn(c)
    assert res.escalated is True
    c.escalation.run.assert_awaited_once()  # the handoff is not delayed
    assert res.reply.startswith("chuyển nhân viên")
    assert res.reply.endswith(ASK_CONTACT_REPLY)
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "collect_contact"
    assert saved.pending_context == {"reason": "user requested human"}


async def test_knowledge_unresolved_asks_for_contact():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="", knowledge_status="no_answer"))
    c.escalation.run = AsyncMock(return_value=_escalation_result("knowledge unresolved"))
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="lỗi lạ", attachments=[]
    )
    assert res.reply.endswith(ASK_CONTACT_REPLY)
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "collect_contact"
    assert saved.pending_context["reason"] == "knowledge unresolved"


async def test_ungrounded_answer_asks_for_contact_after_fallback():
    c = _flagged_coord([{"span": "Tạo mới", "severity": "critical", "reason": "sai nút"}])
    c.escalation.run = AsyncMock(return_value=_escalation_result("ungrounded answer"))
    res = await _turn(c)
    assert res.escalated is True
    assert res.reply.endswith(ASK_CONTACT_REPLY)
    assert "Tạo mới" not in res.reply  # still the fallback text, not the bad answer
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "collect_contact"


def test_the_contact_gate_edits_the_session_the_turn_holds():
    """One session object per turn. `AgentResult.new_session` used to be a second
    channel to the same object, with `_arm_contact_gate` and `_finish` each
    re-deriving which of the two to use; the driver owns it now and the gate writes
    straight to it."""
    c = _coord()
    session = SessionState(conversation_id="cv1")
    result = AgentResult(
        reply="Để nhân viên xem giúp bạn.",
        escalated=True,
        escalation_reason="knowledge unresolved",
    )
    c._arm_contact_gate(result, session)
    assert result.reply == "Để nhân viên xem giúp bạn.\n\n" + ASK_CONTACT_REPLY
    assert session.pending == "collect_contact"
    assert session.pending_context == {"reason": "knowledge unresolved"}


async def test_the_saved_session_is_the_object_the_driver_mutated():
    """The end-to-end half of the same invariant: whatever the steps wrote to the
    session is what reaches Redis."""
    c = _coord()
    session = SessionState(conversation_id="cv1")
    c.sessions.get.return_value = session
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="", knowledge_status="no_answer"))
    c.escalation.run = AsyncMock(return_value=_escalation_result("knowledge unresolved"))
    await _turn(c)
    assert c.sessions.save.call_args.args[0] is session
    assert session.pending == "collect_contact"


async def test_verified_bug_asks_for_contact_and_remembers_the_records():
    c = _verified_bug_coord()
    c.mantis.create_issue.return_value = MantisIssue(
        id=7, url="https://mantis.example/view.php?id=7"
    )
    c.escalation.run = AsyncMock(return_value=_escalation_result("verified bug"))
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="lỗi 500", attachments=[]
    )
    assert res.escalated is True
    assert res.reply.endswith(ASK_CONTACT_REPLY)
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "collect_contact"
    assert saved.pending_context == {
        "reason": "verified bug",
        "backlog_id": "r1",
        "mantis_issue_id": 7,
        "mantis_issue_url": "https://mantis.example/view.php?id=7",
    }


async def test_verified_bug_without_ticket_remembers_only_the_backlog_row():
    c = _verified_bug_coord()
    c.escalation.run = AsyncMock(return_value=_escalation_result("verified bug"))
    await c.handle_turn(customer_id="c1", conversation_id="cv1", message="lỗi 500", attachments=[])
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending_context == {
        "reason": "verified bug",
        "backlog_id": "r1",
        "mantis_issue_id": None,
        "mantis_issue_url": None,
    }


def _awaiting_contact_coord(refs: dict):
    c = _coord()
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1", pending="collect_contact", pending_context=refs
    )
    return c


_BUG_REFS = {
    "reason": "verified bug",
    "backlog_id": "r1",
    "mantis_issue_id": 7,
    "mantis_issue_url": "https://mantis.example/view.php?id=7",
}


async def test_contact_reply_is_attached_everywhere_and_cs_is_told():
    c = _awaiting_contact_coord(_BUG_REFS)
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="0912 345 678, a@b.vn nhé", attachments=[]
    )
    assert res.reply == CONTACT_THANKS_REPLY
    assert res.escalated is False
    c.triage.run.assert_not_called()
    expected = ContactInfo(phone="0912345678", email="a@b.vn", raw="0912 345 678, a@b.vn nhé")
    c.conversations.set_contact.assert_awaited_once_with("cv1", "c1", expected)
    c.backlog.set_contact.assert_awaited_once_with("r1", expected)
    c.mantis.add_note.assert_awaited_once()
    note_args = c.mantis.add_note.call_args.args
    assert note_args[0] == 7 and "0912345678" in note_args[1] and "a@b.vn" in note_args[1]
    kw = c.escalator.contact_update.call_args.kwargs
    assert kw["customer_id"] == "c1" and kw["customer_name"] == "C1"
    assert kw["reason"] == "verified bug"
    assert kw["contact"] == expected
    assert kw["ticket_url"] == "https://mantis.example/view.php?id=7"
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending is None and saved.pending_context is None


async def test_contact_reply_without_records_skips_ticket_and_backlog_updates():
    c = _awaiting_contact_coord({"reason": "user requested human"})
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="sđt 0912345678", attachments=[]
    )
    assert res.reply == CONTACT_THANKS_REPLY
    c.conversations.set_contact.assert_awaited_once()
    c.backlog.set_contact.assert_not_awaited()
    c.mantis.add_note.assert_not_awaited()
    assert c.escalator.contact_update.call_args.kwargs["ticket_url"] is None


async def test_message_without_contact_is_routed_normally_and_updates_nothing():
    c = _awaiting_contact_coord(_BUG_REFS)
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(
        return_value=AgentResult(reply="đáp án mới", knowledge_status="answer")
    )
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="còn cái này thì sao?", attachments=[]
    )
    assert res.reply == "đáp án mới"
    c.triage.run.assert_awaited_once()
    c.conversations.set_contact.assert_not_awaited()
    c.backlog.set_contact.assert_not_awaited()
    c.mantis.add_note.assert_not_awaited()
    c.escalator.contact_update.assert_not_awaited()
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending is None  # asked once, never again


async def test_failed_backlog_update_does_not_stop_the_rest():
    c = _awaiting_contact_coord(_BUG_REFS)
    c.backlog.set_contact.side_effect = RuntimeError("dynamo down")
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="0912345678", attachments=[]
    )
    assert res.reply == CONTACT_THANKS_REPLY
    c.mantis.add_note.assert_awaited_once()
    c.escalator.contact_update.assert_awaited_once()


# ---- direct triage route to issue verification ------------------------------------


async def test_triage_bug_route_starts_verification_without_knowledge():
    """A reported malfunction goes straight to evidence gathering: knowledge is never
    asked, because there is no guide passage that answers "it crashed"."""
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="issue_verification"))
    c.knowledge.run = AsyncMock()
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(reply="gửi ảnh giúp mình", verify_outcome="need_more_info")
    )
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="nhấn Lưu thì báo lỗi 500", attachments=[]
    )
    assert "ảnh" in res.reply
    c.knowledge.run.assert_not_called()
    c.issue_verification.run.assert_awaited_once()
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "verify_issue"


async def test_triage_bug_route_fills_pending_context_from_the_turn():
    """No [[suspected_bug:...]] marker on this route, so the two keys the ticket needs
    come from the turn: the message as summary, the one selected module as a slug."""
    c = _coord()
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1", selected_applications=["Lấy mẫu - Quan trắc"]
    )
    c.conversations.load.return_value = Conversation(
        conversation_id="cv1",
        customer_id="c1",
        turns=[Turn(role="user", content="cũ"), Turn(role="assistant", content="đáp")],
    )
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="issue_verification"))
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(reply="gửi ảnh giúp mình", verify_outcome="need_more_info")
    )
    await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="Import mẫu bị lỗi", attachments=[]
    )
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending_context["summary"] == "Import mẫu bị lỗi"
    assert saved.pending_context["application"] == "lay_mau_quan_trac"
    # index the current (not yet persisted) user turn will get
    assert saved.pending_context["since_turn"] == 2


async def test_triage_bug_route_leaves_application_unset_when_ambiguous():
    """Two selected modules cannot name one ticket, so the field stays None and
    MantisBT renders it as "(không rõ)" rather than guessing wrong."""
    c = _coord()
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1",
        selected_applications=["Lấy mẫu - Quan trắc", "Yêu cầu thử nghiệm"],
    )
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="issue_verification"))
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(reply="gửi ảnh giúp mình", verify_outcome="need_more_info")
    )
    await c.handle_turn(customer_id="c1", conversation_id="cv1", message="bị lỗi", attachments=[])
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending_context["application"] is None


async def test_triage_bug_route_files_a_ticket_like_the_knowledge_route():
    """Downstream cannot tell which route armed the sub-flow: once evidence is complete
    the direct route reaches MantisBT and the backlog exactly as the marker route does."""
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="issue_verification"))
    c.mantis.create_issue.return_value = MantisIssue(id=7, url="https://mantis/7")
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(
            reply="đã đủ",
            verify_outcome="bug_confirmed",
            evidence={
                "application": "lay_mau_quan_trac",
                "summary": "Import mẫu bị lỗi",
                "report": {
                    "title": "Import mẫu báo lỗi",
                    "summary": "Người dùng import mẫu thì hệ thống báo lỗi.",
                    "steps_to_reproduce": "",
                },
            },
        )
    )
    c.escalation.run = AsyncMock(return_value=AgentResult(reply="chuyển nhân viên", escalated=True))
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="Import mẫu bị lỗi", attachments=[]
    )
    assert res.escalated is True
    c.mantis.create_issue.assert_awaited_once()
    c.backlog.add.assert_awaited_once()


# ---- the limits, end to end ----


def _knowledge_coord(*results):
    """A coordinator whose knowledge agent returns each result in turn."""
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(side_effect=list(results))
    c.escalation.run = AsyncMock(return_value=_escalation_result("escalated"))
    return c


async def test_a_clarify_is_counted_on_the_session():
    c = _knowledge_coord(AgentResult(reply="Ý bạn là gì?", knowledge_status="clarify"))
    res = await _turn(c)
    assert res.escalated is False
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending == "knowledge_clarify" and saved.clarify_count == 1


async def test_the_clarify_limit_escalates_instead_of_asking_again():
    """Two questions is the budget. A third would be the bot stalling, so hand over."""
    c = _knowledge_coord(AgentResult(reply="Ý bạn là gì?", knowledge_status="clarify"))
    c.sessions.get.return_value = SessionState(conversation_id="cv1", clarify_count=2)
    res = await _turn(c)
    assert res.escalated is True
    assert c.escalation.run.call_args.kwargs["reason"] == "clarify limit"


async def test_knowledge_is_told_when_its_clarify_budget_is_spent():
    c = _knowledge_coord(AgentResult(reply="đáp án", knowledge_status="answer"))
    c.sessions.get.return_value = SessionState(conversation_id="cv1", clarify_count=2)
    await _turn(c)
    assert c.knowledge.run.call_args.kwargs["allow_clarify"] is False


async def test_the_collection_cap_files_the_ticket_instead_of_asking_forever():
    """Before the cap, an evidence collection that never completed kept `pending`
    set until the Redis TTL dropped it -- the user's only exit was giving up."""
    c = _verified_bug_coord()
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1",
        pending="verify_issue",
        verify_turns=4,
        pending_context={"summary": "A lỗi", "since_turn": 2},
    )
    res = await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="chưa biết nữa", attachments=[]
    )
    assert res.escalated is True
    # Filed without asking the verifier anything more.
    c.issue_verification.run.assert_not_awaited()
    c.mantis.create_issue.assert_awaited_once()
    summary = c.mantis.create_issue.call_args.kwargs["report"].summary
    assert "Thiếu thông tin" in summary


async def test_the_hop_limit_escalates_and_stops_calling_agents():
    c = _coord()
    c.sessions.get.return_value = SessionState(conversation_id="cv1")
    # A knowledge agent that resolves nothing would otherwise be re-entered forever.
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="?"))
    c.escalation.run = AsyncMock(return_value=_escalation_result("hop limit"))
    res = await _turn(c)
    assert res.escalated is True
    assert c.escalation.run.call_args.kwargs["reason"] == "hop limit"
    assert c.knowledge.run.await_count < 6


# ---- user_error: the outcome that costs nothing ----


def _user_error_coord():
    c = _coord()
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="issue_verification"))
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(
            reply="Chức năng này cần chọn mẫu trước khi bấm Lưu.",
            verify_outcome="user_error",
            evidence={"summary": "không lưu được", "since_turn": 0},
        )
    )
    c.knowledge.run = AsyncMock(
        return_value=AgentResult(reply="Anh/Chị chọn mẫu rồi bấm Lưu.", knowledge_status="answer")
    )
    return c


async def test_a_user_error_files_no_ticket_and_pages_nobody():
    c = _user_error_coord()
    c.escalation.run = AsyncMock()
    res = await _turn(c)
    assert res.reply == "Anh/Chị chọn mẫu rồi bấm Lưu."
    assert res.escalated is False
    c.mantis.create_issue.assert_not_awaited()
    c.backlog.add.assert_not_awaited()
    c.escalation.run.assert_not_called()
    c.knowledge.run.assert_awaited_once()


async def test_a_user_error_hands_its_explanation_to_knowledge():
    c = _user_error_coord()
    await _turn(c)
    ctx = c.knowledge.run.call_args.args[0]
    assert ctx.route_hint == "Chức năng này cần chọn mẫu trước khi bấm Lưu."


async def test_a_user_error_closes_the_verification_flow():
    c = _user_error_coord()
    await _turn(c)
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending is None
    assert saved.user_error_seen is True


async def test_knowledge_re_suspecting_a_bug_after_a_user_error_escalates():
    """Knowledge and verification would otherwise hand the turn back and forth,
    paying for a model call each way until the hop limit caught it."""
    c = _coord()
    c.sessions.get.return_value = SessionState(conversation_id="cv1", user_error_seen=True)
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(
        return_value=AgentResult(reply="đáng lẽ chạy", knowledge_status="suspected_bug")
    )
    c.issue_verification.run = AsyncMock()
    c.escalation.run = AsyncMock(return_value=_escalation_result("bug loop"))
    res = await _turn(c)
    assert res.escalated is True
    assert c.escalation.run.call_args.kwargs["reason"] == "bug loop"
    c.issue_verification.run.assert_not_called()


# ---- walking away from a flow ----


async def test_a_cancel_drops_the_pending_flow_and_routes_the_new_topic():
    c = _coord()
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1",
        pending="verify_issue",
        verify_turns=2,
        pending_context={"summary": "A lỗi"},
    )
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="knowledge"))
    c.knowledge.run = AsyncMock(return_value=AgentResult(reply="đáp án", knowledge_status="answer"))
    c.issue_verification.run = AsyncMock()
    res = await c.handle_turn(
        customer_id="c1",
        conversation_id="cv1",
        message="thôi bỏ qua, mình có câu hỏi khác",
        attachments=[],
    )
    assert res.reply == "đáp án"
    c.issue_verification.run.assert_not_called()
    c.triage.run.assert_awaited_once()
    saved = c.sessions.save.call_args.args[0]
    assert saved.pending is None and saved.verify_turns == 0


async def test_an_ordinary_answer_mid_flow_is_not_read_as_a_cancel():
    c = _coord()
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1", pending="verify_issue", pending_context={"summary": "A lỗi"}
    )
    c.triage.run = AsyncMock()
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(reply="còn thiếu gì nữa", verify_outcome="need_more_info")
    )
    await c.handle_turn(
        customer_id="c1", conversation_id="cv1", message="mình chỉ cần in thôi", attachments=[]
    )
    c.issue_verification.run.assert_awaited_once()
    c.triage.run.assert_not_called()


# ---- the route is on the trace ----


async def test_the_route_is_recorded_for_the_trace(monkeypatch):
    c = _knowledge_coord(AgentResult(reply="đáp án", knowledge_status="answer"))
    seen: dict = {}

    class _Span:
        def update(self, **kw):
            seen.update(kw)

    import contextlib

    from agent_customer_support.agents import coordinator as coord_mod

    @contextlib.contextmanager
    def fake_trace(name, **kwargs):
        yield _Span()

    monkeypatch.setattr(coord_mod.tracing, "trace", fake_trace)
    await _turn(c)
    assert seen["output"]["path"] == ["triage", "knowledge", "reply"]


# ---- earlier screenshots are re-shown to the verifier ----


def _resume_coord(turns, since_turn=0):
    c = _coord()
    c.conversations.load.return_value = Conversation(
        conversation_id="cv1", customer_id="c1", turns=turns
    )
    c.sessions.get.return_value = SessionState(
        conversation_id="cv1",
        pending="verify_issue",
        pending_context={"summary": "A lỗi", "since_turn": since_turn},
    )
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(reply="còn thiếu gì", verify_outcome="need_more_info")
    )
    c.attachments.get_bytes = AsyncMock(side_effect=lambda st: st.s3_key.encode())
    return c


async def test_earlier_screenshots_reach_the_verifier_newest_two_only():
    c = _resume_coord(
        [
            Turn(role="user", content="hỏi cũ", attachments=[_stored("old.png")]),
            Turn(role="assistant", content="đáp cũ"),
            Turn(role="user", content="lỗi đây", attachments=[_stored("a.png"), _stored("b.png")]),
            Turn(role="assistant", content="menu nào?"),
            Turn(role="user", content="thêm ảnh", attachments=[_stored("c.png")]),
            Turn(role="assistant", content="còn gì?"),
        ],
        since_turn=2,
    )
    await _turn(c)
    ctx = c.issue_verification.run.call_args.args[0]
    # Read from the start of the conversation, capped to the newest two.
    decoded = [base64.b64decode(a.data).decode() for a in ctx.evidence_images]
    assert decoded == ["b.png", "c.png"]


async def test_a_screenshot_sent_before_the_bug_was_suspected_reaches_the_verifier():
    c = _resume_coord(
        [
            Turn(role="user", content="hỏi cũ", attachments=[_stored("old.png")]),
            Turn(role="assistant", content="đáp cũ"),
            Turn(role="user", content="vẫn lỗi"),
            Turn(role="assistant", content="menu nào?"),
        ],
        since_turn=2,
    )
    await _turn(c)
    ctx = c.issue_verification.run.call_args.args[0]
    assert [base64.b64decode(a.data).decode() for a in ctx.evidence_images] == ["old.png"]


async def test_a_screenshot_that_cannot_be_read_is_dropped_and_the_turn_still_answers():
    c = _resume_coord(
        [Turn(role="user", content="lỗi", attachments=[_stored("a.png"), _stored("b.png")])]
    )

    async def flaky(st):
        if st.s3_key == "a.png":
            raise RuntimeError("s3 down")
        return b"b"

    c.attachments.get_bytes = AsyncMock(side_effect=flaky)
    res = await _turn(c)
    assert res.reply == "còn thiếu gì"
    ctx = c.issue_verification.run.call_args.args[0]
    assert [base64.b64decode(a.data) for a in ctx.evidence_images] == [b"b"]


async def test_the_first_verification_turn_sees_earlier_screenshots():
    c = _coord()
    c.conversations.load.return_value = Conversation(
        conversation_id="cv1",
        customer_id="c1",
        turns=[Turn(role="user", content="hỏi cũ", attachments=[_stored("old.png")])],
    )
    c.triage.run = AsyncMock(return_value=AgentResult(routed_to="issue_verification"))
    c.issue_verification.run = AsyncMock(
        return_value=AgentResult(reply="menu nào?", verify_outcome="need_more_info")
    )
    c.attachments.get_bytes = AsyncMock(side_effect=lambda st: st.s3_key.encode())
    await _turn(c)
    ctx = c.issue_verification.run.call_args.args[0]
    assert [base64.b64decode(a.data).decode() for a in ctx.evidence_images] == ["old.png"]
