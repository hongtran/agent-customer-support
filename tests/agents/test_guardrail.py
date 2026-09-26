import pytest
from unittest.mock import patch
from agent_customer_support.agents.guardrail import (
    GuardrailAgent,
    only_minor,
    apply_claims,
)
from agent_customer_support.llm.schemas import GroundingVerdict, UnsupportedClaim

pytestmark = pytest.mark.asyncio

_SOURCES = ["Vào menu Phiếu yêu cầu, nhấn Tạo mới."]


def _minor(span: str, reason: str = "Nguồn không nêu", replacement: str = "") -> dict:
    return {"span": span, "replacement": replacement, "severity": "minor", "reason": reason}


def _major(span: str, reason: str = "Sai nút") -> dict:
    return {"span": span, "replacement": "", "severity": "major", "reason": reason}


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
                UnsupportedClaim(
                    span="nút Xuất Excel", replacement="", severity="major", reason="bịa nút"
                ),
                UnsupportedClaim(
                    span="Đang sử dụng, ", replacement="", severity="minor", reason="thừa"
                ),
            ],
        ),
    ):
        res = await g.check_output("Đang sử dụng, nhấn nút Xuất Excel ở góc phải.", _SOURCES)
    assert res["pass"] is False
    # The reasons are joined so the log line and the eval CSV still read as one string.
    assert res["reason"] == "bịa nút | thừa"
    assert res["unsupported_claims"] == [
        {"span": "nút Xuất Excel", "replacement": "", "severity": "major", "reason": "bịa nút"},
        {"span": "Đang sử dụng, ", "replacement": "", "severity": "minor", "reason": "thừa"},
    ]


async def test_grounded_reply_passes():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GroundingVerdict(grounded=True, unsupported_claims=[]),
    ):
        res = await g.check_output("Vào Phiếu yêu cầu rồi nhấn Tạo mới.", _SOURCES)
    assert res["pass"] is True


async def test_the_judge_sees_the_customer_question_before_the_sources():
    """The guides never hold the customer's own facts (3 volumes, 2 rooms), so a reply
    that applies a generic step to them is only judgeable with the question in view."""
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GroundingVerdict(grounded=True, unsupported_claims=[]),
    ) as llm:
        await g.check_output(
            "Khai báo 3 thể tích là 3 tham số.", _SOURCES, question="Có 3 thể tích"
        )
    content = llm.call_args.kwargs["messages"][0]["content"]
    assert content.startswith("CÂU HỎI CỦA KHÁCH HÀNG:\nCó 3 thể tích\n\nNGUỒN:")


async def test_no_question_adds_no_header():
    g = GuardrailAgent()
    with patch(
        "agent_customer_support.agents.guardrail.complete_structured",
        return_value=GroundingVerdict(grounded=True, unsupported_claims=[]),
    ) as llm:
        await g.check_output("Vào Phiếu yêu cầu.", _SOURCES)
    assert llm.call_args.kwargs["messages"][0]["content"].startswith("NGUỒN:")


async def test_no_source_passages_skips_the_judge():
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


def test_only_minor_is_false_when_any_claim_is_major():
    assert only_minor([_minor("a"), _major("b")]) is False


def test_only_minor_is_true_when_every_claim_is_minor():
    assert only_minor([_minor("a"), _minor("b")]) is True


# --- apply_claims ----------------------------------------------------------------

_REPLY = (
    "Đang sử dụng, Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu. Sau đó nhấn Lưu."
)


def test_apply_claims_deletes_a_short_minor_span_cleanly():
    out = apply_claims(_REPLY, [_minor("Đang sử dụng, ")])
    assert out == ("Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu. Sau đó nhấn Lưu.")


def test_apply_claims_tidies_the_whitespace_left_behind():
    out = apply_claims("Nhấn Lưu ở góc phải để lưu phiếu yêu cầu.", [_minor(" ở góc phải")])
    assert out == "Nhấn Lưu để lưu phiếu yêu cầu."
    out = apply_claims("Nhấn Lưu (góc phải), rồi thoát màn hình.", [_minor(" (góc phải)")])
    assert out == "Nhấn Lưu, rồi thoát màn hình."


def test_apply_claims_refuses_a_major_claim():
    assert apply_claims(_REPLY, [_major("Đang sử dụng, ")]) is None


def test_apply_claims_refuses_a_span_not_in_the_reply():
    assert apply_claims(_REPLY, [_minor("nút Xuất Excel")]) is None


def test_apply_claims_refuses_a_span_that_occurs_twice():
    assert apply_claims(_REPLY, [_minor("nhấn ")]) is None


def test_apply_claims_refuses_an_empty_span():
    assert apply_claims(_REPLY, [_minor("   ")]) is None


def test_apply_claims_refuses_a_long_span():
    long = "Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu"
    assert len(long) <= 80  # under the char cap: the word cap is what should refuse it
    assert apply_claims(_REPLY, [_minor(long)]) is None
    assert apply_claims("x " * 50 + "y", [_minor("x " * 45)]) is None


def test_apply_claims_refuses_a_whole_sentence():
    """A span ending in sentence punctuation is a full sentence, which is the LLM's job:
    deleting it can silently drop a step the user needed."""
    assert apply_claims(_REPLY, [_minor("Sau đó nhấn Lưu.")]) is None
    assert apply_claims("Xong rồi!\nTiếp theo nhấn Lưu.", [_minor("rồi!\nTiếp")]) is None


def test_apply_claims_is_all_or_nothing():
    """One bad claim refuses the whole delete: a half-repaired reply would ship the
    unsupported text the judge flagged, with no record that it was flagged."""
    assert apply_claims(_REPLY, [_minor("Đang sử dụng, "), _minor("nút Xuất Excel")]) is None


def test_apply_claims_refuses_overlapping_spans():
    """The second span is checked against the text AFTER the first deletion, so a span
    the first one already ate is 'not found' rather than silently skipped."""
    assert apply_claims(_REPLY, [_minor("Đang sử dụng, "), _minor("sử dụng")]) is None


def test_apply_claims_keeps_image_markers_intact():
    reply = "Nhấn [[img:icon:lay_mau/image3.png]] ở góc phải để tạo hồ sơ quan trắc."
    out = apply_claims(reply, [_minor(" ở góc phải")])
    assert out == "Nhấn [[img:icon:lay_mau/image3.png]] để tạo hồ sơ quan trắc."


def test_apply_claims_refuses_a_span_inside_an_image_marker():
    reply = "Nhấn [[img:icon:lay_mau/image3.png]] để tạo hồ sơ."
    assert apply_claims(reply, [_minor("icon:lay_mau")]) is None


def test_apply_claims_refuses_when_the_deletion_guts_the_reply():
    assert apply_claims("Nhấn nút Lưu ở góc phải nhé", [_minor("nút Lưu ở góc phải nhé")]) is None


# --- apply_claims with a replacement ------------------------------------------------

_SIGN = (
    "Sau khi gửi, Anh/Chị theo dõi Mã ký số/trạng thái và kiểm tra thông báo hệ thống "
    "trước khi xử lý tiếp. Khi trạng thái là Đã ký, phiếu được chuyển sang bước duyệt."
)
_SIGN_SPAN = "theo dõi Mã ký số/trạng thái và kiểm tra thông báo hệ thống trước khi xử lý tiếp"


def test_apply_claims_substitutes_the_replacement_for_the_span():
    out = apply_claims(
        _SIGN, [_minor(_SIGN_SPAN, replacement="theo dõi Mã ký số/trạng thái trước khi xử lý tiếp")]
    )
    assert out == (
        "Sau khi gửi, Anh/Chị theo dõi Mã ký số/trạng thái trước khi xử lý tiếp. "
        "Khi trạng thái là Đã ký, phiếu được chuyển sang bước duyệt."
    )


def test_apply_claims_refuses_a_replacement_that_adds_a_word():
    """The whole risk of letting the judge write text: a replacement is only a way to
    keep the sentence grammatical, never a channel for new claims."""
    bad = "theo dõi Mã ký số/trạng thái trên màn hình trước khi xử lý tiếp"
    assert apply_claims(_SIGN, [_minor(_SIGN_SPAN, replacement=bad)]) is None


def test_apply_claims_refuses_a_replacement_that_reorders_the_words():
    bad = "trước khi xử lý tiếp theo dõi Mã ký số/trạng thái"
    assert apply_claims(_SIGN, [_minor(_SIGN_SPAN, replacement=bad)]) is None


def test_apply_claims_refuses_a_replacement_equal_to_the_span():
    """Nothing removed means the flagged claim is still there."""
    assert apply_claims(_SIGN, [_minor(_SIGN_SPAN, replacement=_SIGN_SPAN)]) is None
    assert apply_claims(_SIGN, [_minor(_SIGN_SPAN, replacement=_SIGN_SPAN + ".")]) is None


def test_apply_claims_allows_punctuation_and_case_fixes_in_the_replacement():
    """Deleting a leading clause leaves a lowercase sentence start; fixing that is
    grammar, not content, so the subsequence check compares bare words."""
    reply = "Đang sử dụng, anh/chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu."
    out = apply_claims(reply, [_minor("Đang sử dụng, anh/chị vào", replacement="Anh/chị vào")])
    assert out == "Anh/chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu."
    out = apply_claims(
        "Nhấn Lưu ở góc phải, rồi thoát màn hình.", [_minor("Lưu ở góc phải,", replacement="Lưu,")]
    )
    assert out == "Nhấn Lưu, rồi thoát màn hình."


def test_apply_claims_with_a_replacement_accepts_a_long_span_and_a_full_sentence():
    """The whole-sentence and length guards exist because a pure delete can drop a step.
    With a replacement the sentence survives, so only the amount REMOVED is capped."""
    span = _SIGN_SPAN + "."
    assert len(span.split()) > 12
    out = apply_claims(
        _SIGN, [_minor(span, replacement="theo dõi Mã ký số/trạng thái trước khi xử lý tiếp.")]
    )
    assert out is not None and "kiểm tra thông báo" not in out


def test_apply_claims_refuses_a_replacement_that_removes_too_many_words():
    reply = " ".join(f"w{i}" for i in range(20)) + " cuối cùng nhấn Lưu để hoàn tất phiếu."
    span = " ".join(f"w{i}" for i in range(20))
    assert apply_claims(reply, [_minor(span, replacement="w0 w19")]) is None
    # removing 12 is still Python's job
    assert (
        apply_claims(reply, [_minor(span, replacement=" ".join(f"w{i}" for i in range(8)))])
        is not None
    )


def test_apply_claims_a_pure_delete_is_still_held_to_the_sentence_guard():
    assert apply_claims(_REPLY, [_minor("Sau đó nhấn Lưu.", replacement="")]) is None
