"""Unit tests for the grounding-guardrail eval harness — pure functions only, no LLM calls.

Imported as `eval.guardrail_eval` (like `tests/test_golden_from_excel.py`) rather than by
file path: the module needs `eval.eval` for the shared judge/cost helpers, so the package
is loaded either way.
"""

import csv
from unittest.mock import AsyncMock, MagicMock

import pytest

from eval import guardrail_eval as ge


def _row(
    rid: str,
    judge_label: str = "grounded",
    guardrail_pass: bool = True,
    label: str = "",
    outcome: str = "answered",
    error: str = "",
    reason: str = "",
    unsupported: str = "",
    judge_reason: str = "",
    repair_path: str = "",
    repaired_answer: str = "",
    repaired_guardrail_pass: bool | str = "",
    final_escalated: bool | None = None,
) -> dict:
    """A row shaped like `_one` returns."""
    return {
        "id": rid,
        "question": f"câu hỏi {rid}",
        "application": "Lấy mẫu - Quan trắc",
        "reference_answer": "tham chiếu",
        "outcome": outcome,
        "n_cited": 1,
        "source": "[0] đoạn trích",
        "answer": f"trả lời {rid}",
        "answer_citations": "Mục A",
        "guardrail_pass": guardrail_pass,
        "guardrail_reason": reason,
        "unsupported_claims": unsupported,
        "repair_path": repair_path,
        "repaired_answer": repaired_answer,
        "repaired_guardrail_pass": repaired_guardrail_pass,
        "repaired_unsupported_claims": "",
        "final_escalated": (not guardrail_pass) if final_escalated is None else final_escalated,
        "judge_label": judge_label,
        "judge_reason": judge_reason,
        "judge_extra_claims": "",
        "label": label,
        "latency_s": 1.0,
        "cost_usd": 0.001,
        "guardrail_cost_usd": 0.0005,
        "judge_cost_usd": 0.0002,
        "repair_cost_usd": 0.0,
        "models_used": "{}",
        "error": error,
    }


# --- labels ----------------------------------------------------------------


def test_normalise_label_accepts_the_three_labels_case_insensitively():
    assert ge.normalise_label(" Grounded ") == "grounded"
    assert ge.normalise_label("MINOR") == "minor"
    assert ge.normalise_label("critical") == "critical"


def test_normalise_label_keeps_empty_as_unset():
    assert ge.normalise_label("") == ""
    assert ge.normalise_label(None) == ""


def test_normalise_label_raises_on_typo_naming_the_row():
    with pytest.raises(ValueError, match="Q007"):
        ge.normalise_label("groundd", "Q007")


def test_final_label_prefers_the_human_label():
    assert ge.final_label(_row("Q1", judge_label="critical", label="minor")) == "minor"
    assert ge.final_label(_row("Q1", judge_label="critical")) == "critical"
    assert ge.final_label(_row("Q1", judge_label="")) == ""


# --- metrics ---------------------------------------------------------------


def test_false_escalate_rate_counts_escalated_grounded_and_minor_only():
    rows = [
        _row("Q1", "grounded", guardrail_pass=True),
        _row("Q2", "grounded", guardrail_pass=False),  # false escalate
        _row("Q3", "minor", guardrail_pass=False),  # false escalate
        _row("Q4", "minor", guardrail_pass=True),
        _row("Q5", "critical", guardrail_pass=False),  # correct escalate, not in denominator
    ]
    rate, k, n = ge.false_escalate_rate(rows)
    assert (k, n) == (2, 4)
    assert rate == pytest.approx(0.5)


def test_false_escalate_rate_excludes_unrated_rows():
    rows = [
        _row("Q1", "grounded", guardrail_pass=False, error="RuntimeError: boom"),
        _row("Q2", "grounded", guardrail_pass=False, outcome="clarify"),
        _row("Q3", ge.JUDGE_ERROR, guardrail_pass=False),
        _row("Q4", "", guardrail_pass=False),
        _row("Q5", "grounded", guardrail_pass=True),
    ]
    assert ge.false_escalate_rate(rows) == (0.0, 0, 1)


def test_false_escalate_rate_is_zero_on_an_empty_set():
    assert ge.false_escalate_rate([]) == (0.0, 0, 0)


def test_human_label_changes_the_rate():
    rows = [_row("Q1", judge_label="critical", guardrail_pass=False)]
    assert ge.false_escalate_rate(rows) == (0.0, 0, 0)
    rows[0]["label"] = "minor"
    assert ge.false_escalate_rate(rows) == (1.0, 1, 1)


def test_label_matrix_counts_pass_and_escalate_per_label():
    rows = [
        _row("Q1", "grounded", True),
        _row("Q2", "grounded", False),
        _row("Q3", "minor", False),
        _row("Q4", "critical", True),
        _row("Q5", "critical", True, outcome="no_answer"),  # not an answer: excluded
    ]
    assert ge.label_matrix(rows) == {
        "grounded": {"pass": 1, "escalate": 1},
        "minor": {"pass": 0, "escalate": 1},
        "critical": {"pass": 1, "escalate": 0},
    }


# --- guardrail verdict columns ----------------------------------------------------


def _minor(span: str, reason: str = "thừa", replacement: str = "") -> dict:
    return {"span": span, "replacement": replacement, "severity": "minor", "reason": reason}


def test_format_claims_renders_severity_span_replacement_and_reason_per_claim():
    claims = [
        _minor("Đang sử dụng, ", "không có trong nguồn"),
        {"span": "nút Xuất Excel", "replacement": "", "severity": "critical", "reason": "bịa nút"},
        _minor("Lưu ở góc phải,", "vị trí", replacement="Lưu,"),
    ]
    assert ge.format_claims(claims) == (
        'minor: "Đang sử dụng, " (không có trong nguồn) | critical: "nút Xuất Excel" (bịa nút)'
        ' | minor: "Lưu ở góc phải," => "Lưu," (vị trí)'
    )
    assert ge.format_claims([]) == ""


def test_repair_path_is_python_when_a_replacement_applies():
    verdict = {
        "pass": False,
        "reason": "x",
        "unsupported_claims": [_minor("Đang sử dụng, Anh/Chị", replacement="Anh/Chị")],
    }
    assert ge.repair_path(verdict, _ANSWER) == "python"


_ANSWER = "Đang sử dụng, Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu."


def test_repair_path_is_empty_when_the_guardrail_passed():
    assert ge.repair_path({"pass": True, "reason": ""}, _ANSWER) == ""


def test_repair_path_is_python_when_apply_claims_would_succeed():
    verdict = {"pass": False, "reason": "x", "unsupported_claims": [_minor("Đang sử dụng, ")]}
    assert ge.repair_path(verdict, _ANSWER) == "python"


def test_repair_path_is_llm_when_only_minor_but_python_refuses():
    # A whole sentence: apply_claims refuses, so the LLM repair would run.
    verdict = {"pass": False, "reason": "x", "unsupported_claims": [_minor("để lập phiếu.")]}
    assert ge.repair_path(verdict, _ANSWER) == "llm"


def test_repair_path_is_escalate_on_a_critical_claim_or_no_claims():
    crit = {"span": "Tạo mới", "severity": "critical", "reason": "sai nút"}
    verdict = {"pass": False, "reason": "x", "unsupported_claims": [_minor("Đang sử dụng, "), crit]}
    assert ge.repair_path(verdict, _ANSWER) == "escalate"
    assert (
        ge.repair_path({"pass": False, "reason": "ungrounded", "unsupported_claims": []}, _ANSWER)
        == "escalate"
    )


def test_summarise_counts_the_repair_paths_over_failed_rows():
    rows = [
        _row("Q1", "grounded", True),
        _row("Q2", "minor", False, repair_path="python"),
        _row("Q3", "minor", False, repair_path="python"),
        _row("Q4", "minor", False, repair_path="llm"),
        _row("Q5", "critical", False, repair_path="escalate"),
    ]
    text = ge.summarise(rows)
    assert "python 2" in text and "llm 1" in text and "escalate 1" in text
    # each false escalation names the rung that would have rescued it
    assert "Q2  minor  repair=python" in text


# --- repair outcome: run the rung, judge what it produced ---------------------------


def _patched_agents(monkeypatch, repaired: str | None, recheck_pass: bool):
    """Stub the two agents `repair_outcome` drives; return the instances for asserting."""
    knowledge = MagicMock()
    knowledge.repair = AsyncMock(return_value=repaired)
    guardrail = MagicMock()
    guardrail.check_output = AsyncMock(
        return_value={
            "pass": recheck_pass,
            "reason": "" if recheck_pass else "vẫn thừa",
            "unsupported_claims": [] if recheck_pass else [_minor("x")],
        }
    )
    monkeypatch.setattr(ge, "KnowledgeAgent", lambda: knowledge)
    monkeypatch.setattr(ge, "GuardrailAgent", lambda: guardrail)
    return knowledge, guardrail


async def test_repair_outcome_is_empty_when_the_guardrail_passed(monkeypatch):
    knowledge, guardrail = _patched_agents(monkeypatch, "unused", True)
    cols, _cost = await ge.repair_outcome({"pass": True, "reason": ""}, _ANSWER, ["p"])
    assert cols == {
        "repair_path": "",
        "repaired_answer": "",
        "repaired_guardrail_pass": "",
        "repaired_unsupported_claims": "",
        "final_escalated": False,
    }
    knowledge.repair.assert_not_awaited()
    guardrail.check_output.assert_not_awaited()


async def test_python_path_writes_the_stripped_answer_and_judges_it(monkeypatch):
    """Production ships a Python delete without a second judge call; the harness judges
    it anyway, because that is the only way to check the trust-the-delete decision."""
    knowledge, guardrail = _patched_agents(monkeypatch, "unused", True)
    verdict = {"pass": False, "reason": "x", "unsupported_claims": [_minor("Đang sử dụng, ")]}
    cols, _cost = await ge.repair_outcome(verdict, _ANSWER, ["p"])
    assert cols["repair_path"] == "python"
    assert cols["repaired_answer"] == (
        "Anh/Chị vào menu Phiếu yêu cầu rồi nhấn Tạo mới để lập phiếu."
    )
    assert cols["repaired_guardrail_pass"] is True
    assert cols["final_escalated"] is False
    knowledge.repair.assert_not_awaited()
    guardrail.check_output.assert_awaited_once_with(cols["repaired_answer"], ["p"])


async def test_python_path_ships_even_when_the_harness_re_judge_still_flags_it(monkeypatch):
    _patched_agents(monkeypatch, "unused", False)
    verdict = {"pass": False, "reason": "x", "unsupported_claims": [_minor("Đang sử dụng, ")]}
    cols, _cost = await ge.repair_outcome(verdict, _ANSWER, ["p"])
    assert cols["repaired_guardrail_pass"] is False
    assert 'minor: "x"' in cols["repaired_unsupported_claims"]
    # production does not re-judge, so this row is NOT an escalation
    assert cols["final_escalated"] is False


async def test_llm_path_calls_repair_and_follows_the_recheck(monkeypatch):
    knowledge, guardrail = _patched_agents(monkeypatch, "Anh/Chị nhấn Tạo mới.", True)
    claims = [_minor("để lập phiếu.")]
    verdict = {"pass": False, "reason": "x", "unsupported_claims": claims}
    cols, _cost = await ge.repair_outcome(verdict, _ANSWER, ["p"])
    assert cols["repair_path"] == "llm"
    assert cols["repaired_answer"] == "Anh/Chị nhấn Tạo mới."
    assert cols["repaired_guardrail_pass"] is True
    assert cols["final_escalated"] is False
    knowledge.repair.assert_awaited_once_with(_ANSWER, claims, ["p"])
    guardrail.check_output.assert_awaited_once_with("Anh/Chị nhấn Tạo mới.", ["p"])


async def test_llm_path_that_fails_the_recheck_is_a_final_escalation(monkeypatch):
    _patched_agents(monkeypatch, "vẫn sai", False)
    verdict = {"pass": False, "reason": "x", "unsupported_claims": [_minor("để lập phiếu.")]}
    cols, _cost = await ge.repair_outcome(verdict, _ANSWER, ["p"])
    assert cols["repaired_answer"] == "vẫn sai"
    assert cols["repaired_guardrail_pass"] is False
    assert cols["final_escalated"] is True


async def test_llm_path_with_no_repaired_text_is_a_final_escalation(monkeypatch):
    _knowledge, guardrail = _patched_agents(monkeypatch, None, True)
    verdict = {"pass": False, "reason": "x", "unsupported_claims": [_minor("để lập phiếu.")]}
    cols, _cost = await ge.repair_outcome(verdict, _ANSWER, ["p"])
    assert cols["repaired_answer"] == ""
    assert cols["repaired_guardrail_pass"] == ""
    assert cols["final_escalated"] is True
    guardrail.check_output.assert_not_awaited()


async def test_escalate_path_runs_nothing(monkeypatch):
    knowledge, guardrail = _patched_agents(monkeypatch, "unused", True)
    crit = {"span": "Tạo mới", "severity": "critical", "reason": "sai nút"}
    cols, cost = await ge.repair_outcome(
        {"pass": False, "reason": "x", "unsupported_claims": [crit]}, _ANSWER, ["p"]
    )
    assert cols["repair_path"] == "escalate"
    assert cols["final_escalated"] is True
    assert cols["repaired_answer"] == ""
    assert cost.n_calls == 0
    knowledge.repair.assert_not_awaited()
    guardrail.check_output.assert_not_awaited()


# --- after-repair metric --------------------------------------------------------


def test_false_escalate_rate_after_repair_counts_only_final_escalations():
    rows = [
        _row("Q1", "grounded", True),
        _row("Q2", "minor", False, repair_path="python", final_escalated=False),
        _row("Q3", "minor", False, repair_path="llm", final_escalated=False),
        _row("Q4", "minor", False, repair_path="llm", final_escalated=True),
        _row("Q5", "grounded", False, repair_path="escalate", final_escalated=True),
        _row("Q6", "critical", False, repair_path="escalate", final_escalated=True),
    ]
    assert ge.false_escalate_rate(rows) == (pytest.approx(0.8), 4, 5)
    assert ge.false_escalate_rate_after_repair(rows) == (pytest.approx(0.4), 2, 5)


def test_summarise_prints_the_after_repair_rate_and_the_python_rejudge_count():
    rows = [
        _row(
            "Q1",
            "minor",
            False,
            repair_path="python",
            final_escalated=False,
            repaired_answer="a",
            repaired_guardrail_pass=True,
        ),
        _row(
            "Q2",
            "minor",
            False,
            repair_path="python",
            final_escalated=False,
            repaired_answer="b",
            repaired_guardrail_pass=False,
        ),
        _row(
            "Q3",
            "minor",
            False,
            repair_path="llm",
            final_escalated=True,
            repaired_answer="c",
            repaired_guardrail_pass=False,
        ),
    ]
    text = ge.summarise(rows)
    assert "FALSE ESCALATE RATE   100.00%" in text
    assert "AFTER REPAIR" in text and "33.33%" in text
    # 1 of the 2 python deletes would still be flagged by the judge
    assert "python deletes the judge still flags   1 / 2" in text


def test_load_scored_restores_the_repair_columns(tmp_path):
    rows = [
        _row("Q1", "grounded", True),
        _row(
            "Q2",
            "minor",
            False,
            repair_path="python",
            final_escalated=False,
            repaired_answer="a",
            repaired_guardrail_pass=False,
        ),
        _row("Q3", "minor", False, repair_path="escalate", final_escalated=True),
    ]
    path = tmp_path / "g.csv"
    ge.write_rows(rows, path)
    back = ge.load_scored(path)
    assert back[0]["repaired_guardrail_pass"] == ""
    assert back[0]["final_escalated"] is False
    assert back[1]["repaired_guardrail_pass"] is False
    assert back[1]["final_escalated"] is False
    assert back[2]["final_escalated"] is True
    assert ge.false_escalate_rate_after_repair(back) == (pytest.approx(1 / 3), 1, 3)


# --- judge parsing -----------------------------------------------------------


def test_parse_label_reply_reads_a_valid_json_object():
    got = ge.parse_label_reply(
        '{"label": "Minor", "reason": "thêm chi tiết nhỏ", "extra_claims": ["nút Lưu"]}'
    )
    assert got == {"label": "minor", "reason": "thêm chi tiết nhỏ", "extra_claims": ["nút Lưu"]}


def test_parse_label_reply_flags_an_unknown_label():
    got = ge.parse_label_reply('{"label": "ok", "reason": "x", "extra_claims": []}')
    assert got["label"] == ge.JUDGE_ERROR
    assert "ok" in got["reason"]


def test_parse_label_reply_flags_garbage():
    got = ge.parse_label_reply("no json here")
    assert got["label"] == ge.JUDGE_ERROR
    assert got["extra_claims"] == []


# --- report --------------------------------------------------------------------


def test_summarise_prints_each_false_escalation_with_both_reasons():
    rows = [
        _row("Q1", "grounded", True),
        _row(
            "Q2",
            "minor",
            False,
            reason="bịa nút Xuất Excel",
            unsupported="nhấn nút Xuất Excel",
            judge_reason="thêm chi tiết nhỏ",
        ),
    ]
    text = ge.summarise(rows)
    assert "FALSE ESCALATE RATE" in text
    assert "1 / 2" in text
    assert "FALSE ESCALATIONS (1)" in text
    assert "Q2" in text and "bịa nút Xuất Excel" in text and "thêm chi tiết nhỏ" in text
    assert "nhấn nút Xuất Excel" in text


def test_summarise_says_so_when_nothing_was_falsely_escalated():
    text = ge.summarise([_row("Q1", "grounded", True)])
    assert "FALSE ESCALATIONS (0)" in text
    assert "none" in text


def test_summarise_reports_critical_answers_as_a_count_only():
    rows = [
        _row("Q1", "critical", True, judge_reason="sai thứ tự bước"),
        _row("Q2", "critical", False),
    ]
    text = ge.summarise(rows)
    assert "critical passed" in text
    assert "CRITICAL ANSWERS (2)" in text
    assert "sai thứ tự bước" in text
    assert "missed error" not in text.lower() or "not tracked" in text


def test_summarise_counts_errors_and_non_answers_separately():
    rows = [
        _row("Q1", "grounded", True),
        _row("Q2", "", True, outcome="clarify"),
        _row("Q3", "", True, error="RuntimeError: boom"),
    ]
    text = ge.summarise(rows)
    assert "clarify 1" in text
    assert "errors 1" in text


# --- --score loader ------------------------------------------------------------


def test_load_scored_round_trips_the_csv(tmp_path):
    rows = [
        _row("Q1", "grounded", True),
        _row("Q2", "critical", False, label="minor"),
    ]
    path = tmp_path / "guardrail-x.csv"
    ge.write_rows(rows, path)
    back = ge.load_scored(path)
    assert [r["id"] for r in back] == ["Q1", "Q2"]
    assert back[0]["guardrail_pass"] is True
    assert back[1]["guardrail_pass"] is False
    assert back[0]["label"] == ""
    assert back[1]["label"] == "minor"
    assert ge.false_escalate_rate(back) == (0.5, 1, 2)


def test_load_scored_raises_on_a_mislabelled_row(tmp_path):
    path = tmp_path / "g.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(_row("Q1").keys()))
        w.writeheader()
        w.writerow(_row("Q9", label="groundd"))
    with pytest.raises(ValueError, match="Q9"):
        ge.load_scored(path)


# --- judge call ---------------------------------------------------------------------


def test_judge_sends_the_process_block_like_the_guardrail_does(monkeypatch):
    """The agent answers from PROCESS_BLOCK as well as the cited passages, so the judge
    must see it too -- otherwise every process-derived sentence is labelled "minor"."""
    from agent_customer_support.agents.prompts import PROCESS_BLOCK
    from eval.eval import AnswerRun
    from eval.testset import TestQuestion

    cap: dict = {}

    def fake_complete_text(messages, system=None, model=None):
        cap["system"] = system
        cap["content"] = messages[0]["content"]
        return '{"label": "grounded", "reason": "", "extra_claims": []}'

    monkeypatch.setattr(ge, "complete_text", fake_complete_text)
    test = TestQuestion(
        id="Q1", question="q", keywords=[], reference_answer="ref", category="how_to"
    )
    run = AnswerRun(answer="a", outcome="answered", abstained=False, cited_passages=["p0"])

    judged, _cost = ge._judge_label(test, run)

    assert judged["label"] == "grounded"
    assert cap["system"][0] is PROCESS_BLOCK
    assert cap["system"][-1]["text"] == ge._LABEL_JUDGE_SYSTEM
    assert "[0] p0" in cap["content"]
