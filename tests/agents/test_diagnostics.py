from agent_customer_support.agents.diagnostics import (
    DIAGNOSTIC_RULES,
    RULES_BY_ID,
    DiagnosticRule,
)


def test_rules_have_unique_nonempty_fields():
    ids = [r.id for r in DIAGNOSTIC_RULES]
    assert len(ids) == len(set(ids)), "rule ids must be unique"
    for r in DIAGNOSTIC_RULES:
        assert isinstance(r, DiagnosticRule)
        assert r.id and r.symptom and r.guidance


def test_rules_by_id_roundtrip():
    for r in DIAGNOSTIC_RULES:
        assert RULES_BY_ID[r.id] is r
    # the two confirmed core examples must exist
    assert "missing_master_data" in RULES_BY_ID
    assert "no_permission" in RULES_BY_ID
    assert "ui_not_configured" in RULES_BY_ID
