"""Unit tests for the triage eval harness — pure functions only, no LLM calls.

The module is loaded by file path rather than as `eval.triage_eval` on purpose:
importing the `eval` package runs `eval/__init__.py`, which calls `load_dotenv()`.
Collecting this test would then pull the developer's real `.env` (Langfuse keys and
all) into the environment of every other test in the suite.
"""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "eval" / "triage_eval.py"
_spec = importlib.util.spec_from_file_location("triage_eval_under_test", _PATH)
assert _spec and _spec.loader
te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(te)


def _row(rid: str, expected: str, got: str, error: str = "", note: str = "") -> dict:
    """A row shaped like `_run_case` returns."""
    return {
        "id": rid,
        "question": f"câu hỏi {rid}",
        "expected": expected,
        "note": note,
        "got": got,
        "correct": got == expected and not error,
        "error": error,
        "latency_s": 0.5,
    }


# --- loader ---------------------------------------------------------------


def test_load_cases_reads_bom_file_and_normalises_labels(tmp_path):
    csv_file = tmp_path / "t.csv"
    csv_file.write_text(
        "id,question,expected,note\n"
        "T001,Cách tạo phiếu?,knowledge,\n"
        "T002,Cho tôi gặp nhân viên,ESCALATE ,muốn gặp người thật\n"
        "T003,Tỷ giá USD?,blocked,\n",
        encoding="utf-8-sig",  # Excel's BOM: the case the loader exists to survive
    )
    cases = te.load_cases(str(csv_file))
    assert [c.id for c in cases] == ["T001", "T002", "T003"]
    # whitespace/case normalised, and the legacy `blocked` mapped onto out_of_scope
    assert [c.expected for c in cases] == ["knowledge", "escalate", "out_of_scope"]
    assert cases[1].note == "muốn gặp người thật"


def test_load_cases_skips_rows_with_no_question(tmp_path):
    csv_file = tmp_path / "t.csv"
    csv_file.write_text(
        "id,question,expected,note\nT001,Cách tạo phiếu?,knowledge,\nT002,  ,knowledge,\n",
        encoding="utf-8",
    )
    assert [c.id for c in te.load_cases(str(csv_file))] == ["T001"]


def test_load_cases_raises_on_unknown_label_naming_the_row(tmp_path):
    csv_file = tmp_path / "t.csv"
    csv_file.write_text(
        "id,question,expected,note\nT009,Cách tạo phiếu?,konwledge,\n", encoding="utf-8"
    )
    # A typo must fail loudly: scored, it would read as a permanent model failure.
    with pytest.raises(ValueError, match="T009"):
        te.load_cases(str(csv_file))


# --- metrics --------------------------------------------------------------


def test_accuracy_ignores_errored_rows():
    rows = [
        _row("T1", "knowledge", "knowledge"),
        _row("T2", "escalate", "knowledge"),
        _row("T3", "out_of_scope", "", error="APIError: boom"),
    ]
    # 1 of the 2 rows that produced a decision; the errored row is neither right nor wrong
    assert te.accuracy(rows) == pytest.approx(0.5)
    assert len(te.scored(rows)) == 2


def test_confusion_counts_expected_by_got():
    rows = [
        _row("T1", "knowledge", "knowledge"),
        _row("T2", "knowledge", "escalate"),
        _row("T3", "out_of_scope", "knowledge"),
        _row("T4", "out_of_scope", "out_of_scope"),
    ]
    matrix = te.confusion(rows)
    assert matrix["knowledge"] == {"knowledge": 1, "escalate": 1, "out_of_scope": 0}
    assert matrix["out_of_scope"] == {"knowledge": 1, "escalate": 0, "out_of_scope": 1}


def test_confusion_surfaces_an_unexpected_target_as_its_own_column():
    """`flow` is unreachable here — if it ever appears it must be visible, not folded in."""
    rows = [_row("T1", "knowledge", "flow")]
    assert "flow" in te.got_labels(rows)
    assert te.confusion(rows)["knowledge"]["flow"] == 1


def test_per_class_reports_zero_precision_for_a_never_predicted_label():
    rows = [
        _row("T1", "out_of_scope", "knowledge"),
        _row("T2", "out_of_scope", "knowledge"),
    ]
    stats = te.per_class(rows)
    # no prediction of out_of_scope at all: precision is 0.0, not a ZeroDivisionError
    assert stats["out_of_scope"]["precision"] == 0.0
    assert stats["out_of_scope"]["recall"] == 0.0
    assert stats["out_of_scope"]["f1"] == 0.0
    assert stats["out_of_scope"]["support"] == 2
    assert stats["knowledge"]["predicted"] == 2


def test_per_class_f1_on_a_mixed_run():
    rows = [
        _row("T1", "knowledge", "knowledge"),
        _row("T2", "knowledge", "knowledge"),
        _row("T3", "escalate", "knowledge"),
        _row("T4", "escalate", "escalate"),
    ]
    k = te.per_class(rows)["knowledge"]
    assert k["precision"] == pytest.approx(2 / 3)
    assert k["recall"] == pytest.approx(1.0)
    assert k["f1"] == pytest.approx(0.8)


def test_macro_f1_ignores_labels_with_no_rows():
    """A perfect run over two labels scores 1.0 — the absent third must not average in."""
    rows = [
        _row("T1", "knowledge", "knowledge"),
        _row("T2", "escalate", "escalate"),
    ]
    assert te.macro_f1(rows) == pytest.approx(1.0)


# --- report ---------------------------------------------------------------


def test_summarise_prints_the_mismatch_with_its_question_and_note():
    rows = [
        _row("T1", "knowledge", "knowledge"),
        _row("T2", "out_of_scope", "knowledge", note="hỏi về thời tiết"),
    ]
    report = te.summarise(rows)
    assert "MISMATCHES (1)" in report
    assert "T2  out_of_scope -> knowledge" in report
    assert "câu hỏi T2" in report
    assert "hỏi về thời tiết" in report
    # the matching row must not be listed
    assert "T1  " not in report


def test_summarise_says_so_when_nothing_mismatched():
    report = te.summarise([_row("T1", "knowledge", "knowledge")])
    assert "MISMATCHES (0)" in report


def test_summarise_reports_errors_separately():
    rows = [_row("T1", "knowledge", "", error="APIError: boom")]
    report = te.summarise(rows)
    assert "errors" in report
    assert "1 rows" in report
