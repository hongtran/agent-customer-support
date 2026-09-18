import pytest
from unittest.mock import patch
from agent_customer_support.agents.guardrail import (
    GuardrailAgent,
    only_minor,
    strip_claims,
)
from agent_customer_support.llm.schemas import GroundingVerdict, UnsupportedClaim

pytestmark = pytest.mark.asyncio

_SOURCES = ["Vào menu Phiếu yêu cầu, nhấn Tạo mới."]


def _minor(span: str, reason: str = "Nguồn không nêu") -> dict:
    return {"span": span, "severity": "minor", "reason": reason}


def _critical(span: str, reason: str = "Sai nút") -> dict:
    return {"span": span, "severity": "critical", "reason": reason}


async def test_empty_input_blocked():
    g = GuardrailAgent()
    res = await g.check_input("   ")
    assert res["pass"] is False


async def test_oversized_input_blocked():
    g = GuardrailAgent()
    res = await g.check_input("x" * 6000)
    assert res["pass"] is False


async def test_normal_input_passes():
    g = GuardrailAgent()
    res = await g.check_input("làm sao tạo phiếu yêu cầu?")
    assert res["pass"] is True


async def test_ungrounded_reply_is_flagged_with_per_claim_severity():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GroundingVerdict(
            grounded=False,
            unsupported_claims=[
                UnsupportedClaim(span="nút Xuất Excel", severity="critical", reason="bịa nút"),
                UnsupportedClaim(span="Đang sử dụng, ", severity="minor", reason="thừa"),
            ],
        ),
    ):
        res = await g.check_output("Đang sử dụng, nhấn nút Xuất Excel ở góc phải.", _SOURCES)
    assert res["pass"] is False
    # The reasons are joined so the log line and the eval CSV still read as one string.
    assert res["reason"] == "bịa nút | thừa"
    assert res["unsupported_claims"] == [
        {"span": "nút Xuất Excel", "severity": "critical", "reason": "bịa nút"},
        {"span": "Đang sử dụng, ", "severity": "minor", "reason": "thừa"},
    ]


async def test_grounded_reply_passes():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GroundingVerdict(grounded=True, unsupported_claims=[]),
    ):
        res = await g.check_output("Vào Phiếu yêu cầu rồi nhấn Tạo mới.", _SOURCES)
    assert res["pass"] is True


async def test_no_cited_passages_skips_the_judge():
    """Nothing to judge against, so no call at all.

    This is the common case, not an edge one: every non-knowledge route and every
    clarify/no-answer reply arrives with an empty list. Calling the judge there would
    flag correct replies for lacking sources they never claimed to have — and would
    spend a model call on every single turn.
    """
    g = GuardrailAgent()
    with patch("agent_customer_support.agents.guardrail.complete_structured") as llm:
        res = await g.check_output("Bạn đang muốn tạo loại phiếu nào?", [])
    assert res["pass"] is True
    llm.assert_not_called()


async def test_fails_open_when_no_verdict():
    """A judge that could not answer must never silence an already-paid-for reply.
    Deliberately the opposite direction from triage's fail-safe."""
    g = GuardrailAgent()
    with patch("agent_customer_support.agents.guardrail.complete_structured", return_value=None):
        res = await g.check_output("câu trả lời hợp lệ", _SOURCES)
    assert res["pass"] is True
    assert res["reason"] == ""


async def test_flagged_with_no_claims_gets_placeholder_reason():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GroundingVerdict(grounded=False, unsupported_claims=[]),
    ):
        res = await g.check_output("nội dung đáng ngờ", _SOURCES)
    assert res["pass"] is False
    assert res["reason"] == "ungrounded"
    assert res["unsupported_claims"] == []


# --- only_minor ----------------------------------------------------------------


def test_only_minor_is_false_on_an_empty_list():
    """grounded=false with nothing named cannot be repaired — there is nothing to delete."""
    assert only_minor([]) is False


def test_only_minor_is_false_when_any_claim_is_critical():
    assert only_minor([_minor("a"), _critical("b")]) is False


def test_only_minor_is_true_when_every_claim_is_minor():
    assert only_minor([_minor("a"), _minor("b")]) is True


# --- strip_claims ----------------------------------------------------------------

_REPLY = (
    "Đang sử dụng, Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu. Sau đó nhấn Lưu."
)


def test_strip_claims_deletes_a_short_minor_span_cleanly():
    out = strip_claims(_REPLY, [_minor("Đang sử dụng, ")])
    assert out == ("Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu. Sau đó nhấn Lưu.")


def test_strip_claims_tidies_the_whitespace_left_behind():
    out = strip_claims("Nhấn Lưu ở góc phải để lưu phiếu yêu cầu.", [_minor(" ở góc phải")])
    assert out == "Nhấn Lưu để lưu phiếu yêu cầu."
    out = strip_claims("Nhấn Lưu (góc phải), rồi thoát màn hình.", [_minor(" (góc phải)")])
    assert out == "Nhấn Lưu, rồi thoát màn hình."


def test_strip_claims_refuses_a_critical_claim():
    assert strip_claims(_REPLY, [_critical("Đang sử dụng, ")]) is None


def test_strip_claims_refuses_a_span_not_in_the_reply():
    assert strip_claims(_REPLY, [_minor("nút Xuất Excel")]) is None


def test_strip_claims_refuses_a_span_that_occurs_twice():
    assert strip_claims(_REPLY, [_minor("nhấn ")]) is None


def test_strip_claims_refuses_an_empty_span():
    assert strip_claims(_REPLY, [_minor("   ")]) is None


def test_strip_claims_refuses_a_long_span():
    long = "Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu"
    assert len(long) <= 80  # under the char cap: the word cap is what should refuse it
    assert strip_claims(_REPLY, [_minor(long)]) is None
    assert strip_claims("x " * 50 + "y", [_minor("x " * 45)]) is None


def test_strip_claims_refuses_a_whole_sentence():
    """A span ending in sentence punctuation is a full sentence, which is the LLM's job:
    deleting it can silently drop a step the user needed."""
    assert strip_claims(_REPLY, [_minor("Sau đó nhấn Lưu.")]) is None
    assert strip_claims("Xong rồi!\nTiếp theo nhấn Lưu.", [_minor("rồi!\nTiếp")]) is None


def test_strip_claims_is_all_or_nothing():
    """One bad claim refuses the whole delete: a half-repaired reply would ship the
    unsupported text the judge flagged, with no record that it was flagged."""
    assert strip_claims(_REPLY, [_minor("Đang sử dụng, "), _minor("nút Xuất Excel")]) is None


def test_strip_claims_refuses_overlapping_spans():
    """The second span is checked against the text AFTER the first deletion, so a span
    the first one already ate is 'not found' rather than silently skipped."""
    assert strip_claims(_REPLY, [_minor("Đang sử dụng, "), _minor("sử dụng")]) is None


def test_strip_claims_keeps_image_markers_intact():
    reply = "Nhấn [[img:icon:lay_mau/image3.png]] ở góc phải để tạo hồ sơ quan trắc."
    out = strip_claims(reply, [_minor(" ở góc phải")])
    assert out == "Nhấn [[img:icon:lay_mau/image3.png]] để tạo hồ sơ quan trắc."


def test_strip_claims_refuses_a_span_inside_an_image_marker():
    reply = "Nhấn [[img:icon:lay_mau/image3.png]] để tạo hồ sơ."
    assert strip_claims(reply, [_minor("icon:lay_mau")]) is None


def test_strip_claims_refuses_when_the_deletion_guts_the_reply():
    assert strip_claims("Nhấn nút Lưu ở góc phải nhé", [_minor("nút Lưu ở góc phải nhé")]) is None
