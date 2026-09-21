"""Evaluate the output guardrail against what KnowledgeAgent actually writes.

    poetry run python -m eval.guardrail_eval                         # all rows
    poetry run python -m eval.guardrail_eval --limit 5               # smoke test
    poetry run python -m eval.guardrail_eval --guardrail-model gpt-5.4-mini
    poetry run python -m eval.guardrail_eval --score eval/results/guardrail-<ts>.csv

The output guardrail (`GuardrailAgent.check_output`) flags a reply when a claim in it
is not supported by the passages the reply cited. KnowledgeAgent sometimes adds a few
words that are not in the source -- a harmless tip, a bit of context -- and a flag used
to throw the whole answer away. The coordinator now repairs a reply whose claims are all
minor (a Python delete, then one LLM repair) before escalating; the `repair_path` column
here says which rung each verdict would reach, with no extra LLM call. Nothing measured
any of this until now: the answer harness (`eval/run_eval.py`) never runs the guardrail,
and the triage harness never runs KnowledgeAgent.

Each row runs the real KnowledgeAgent, then the real guardrail on its reply and cited
passages -- the same two calls `Coordinator.handle_turn` makes -- and an LLM judge
labels the answer against the sources and the reference answer. "Sources" means both
the cited passages and `PROCESS_BLOCK` (the QUY TRÌNH text the compose step and the
guardrail always carry in their system prompt); the judge gets the same block, so a
claim the agent took from the process rules is grounded, not extra.

  * grounded  every claim is in a source (paraphrase is fine)
  * minor     an extra detail no source contains, but it changes nothing the user
              does and is not wrong
  * critical  a wrong or invented step, feature, button, number or condition

The headline number is the FALSE ESCALATE RATE: how often a grounded or minor answer
gets escalated. Critical answers that pass are printed as a count only; the missed
error rate is deliberately not tracked yet, because the answerable test set produces
too few critical answers for that number to mean anything.

Labels are the judge's, with a human override: the per-row CSV carries an empty
`label` column, and `--score <csv>` re-reads a file after someone filled it in and
recomputes the report with no LLM or Qdrant call. `label` wins over `judge_label`
wherever both are set.

Needs real Qdrant + embeddings + the provider keys (`.env` via `eval/__init__.py`).
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from agent_customer_support.agents.guardrail import GuardrailAgent, only_minor, apply_claims
from agent_customer_support.agents.knowledge import KnowledgeAgent
from agent_customer_support.agents.prompts import PROCESS_BLOCK
from agent_customer_support.config import get_settings
from agent_customer_support.llm import complete_text, usage
from agent_customer_support.rag_client import RagClient
from eval.eval import (
    AnswerRun,
    TurnCost,
    _extract_json,
    _turn_cost,
    answer_question,
    fmt_usd,
    format_citation,
    judge_model,
)
from eval.testset import TEST_FILE, TestQuestion, load_tests

RESULTS_DIR = Path(__file__).parent / "results"

LABELS = ("grounded", "minor", "critical")

# Sentinel for a judge reply that could not be read. Kept outside `LABELS` so a broken
# judge shows up as its own count in the report instead of being scored either way.
JUDGE_ERROR = "JUDGE_ERROR"

# Same set as `run_eval._MODEL_ENV_VARS`: `--model` pins the whole pipeline to one model
# so an A/B compares like with like. `--guardrail-model` sets only the last one.
_MODEL_ENV_VARS = (
    "AGENT_MODEL",
    "TRIAGE_MODEL",
    "KNOWLEDGE_MODEL",
    "KNOWLEDGE_CONTEXTUALIZE_MODEL",
    "VERIFICATION_MODEL",
    "FLOW_MODEL",
    "GUARDRAIL_MODEL",
)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def normalise_label(raw: str | None, row_id: str = "?") -> str:
    """Canonical label, `""` for unset, or raise naming the row.

    Raising at load time is the point: a typo ("groundd") in the human column would
    otherwise silently drop the row from the rate and read as a smaller denominator.
    """
    label = (raw or "").strip().casefold()
    if not label:
        return ""
    if label not in LABELS:
        raise ValueError(f"row {row_id}: label {raw!r} is not one of {', '.join(LABELS)}")
    return label


def final_label(row: dict) -> str:
    """The label that scores: the human one when filled, else the judge's."""
    return row.get("label") or row.get("judge_label") or ""


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------

# English instructions, Vietnamese reason -- the same convention as `eval.eval._JUDGE_SYSTEM`.
# The exclusions mirror `GROUNDING_JUDGE_PROMPT` so the judge does not call "minor" what
# the guardrail is explicitly told to ignore.
#
# The judge is sent with `PROCESS_BLOCK` ahead of it, exactly as the compose step and the
# guardrail are: the agent may answer from the process rules (QUY TRÌNH) as well as from the
# cited passages, so a claim that comes from there is grounded, not extra. Judging against
# the passages alone would label every process-derived sentence "minor".
_LABEL_JUDGE_SYSTEM = """You are an expert evaluator for a Vietnamese-language customer support \
assistant for the CENLAB laboratory software. The assistant answers from TWO sources: the \
QUY TRÌNH (process) text at the top of this system prompt, which it always has, and the SOURCE \
passages it cited for this answer. You receive a question, the SOURCE passages, a REFERENCE \
answer written by support staff, and the GENERATED answer.

Classify the generated answer with exactly one label:
- "grounded": every claim about steps, buttons, menus, screens, order, conditions, numbers \
or permissions is stated in the SOURCE or in the QUY TRÌNH, or follows from them. Paraphrase \
is fine. Greetings, lead-ins, clarifying questions, stated assumptions, "contact your system \
admin" routing lines, and image tokens like [[img:screen:...]] are NOT extra claims.
- "minor": the answer adds a small detail that is in NEITHER the SOURCE nor the QUY TRÌNH but \
is not wrong and does not change what the user would do (a generic tip, harmless context, an \
obvious UI generality). Use the REFERENCE answer to decide whether the extra detail is \
actually wrong.
- "critical": the answer contains a wrong or invented step or step order, a wrong feature, \
button, menu or screen name, a wrong number, condition or permission, or contradicts the \
SOURCE, the QUY TRÌNH or the REFERENCE.

List every claim that is in neither the SOURCE nor the QUY TRÌNH in extra_claims, quoted \
from the answer.

Reply with ONLY a JSON object, no markdown fence and no prose around it:
{"label": "grounded" | "minor" | "critical", "reason": "<one or two sentences in Vietnamese>", \
"extra_claims": ["<claim>", ...]}"""


def parse_label_reply(text: str | None) -> dict:
    """Read the judge's JSON into `{label, reason, extra_claims}`.

    Anything unreadable, or a label outside `LABELS`, becomes `JUDGE_ERROR` with the raw
    text in `reason` so the row can be re-judged by hand.
    """
    raw = _extract_json(text or "")
    if raw is None:
        return {
            "label": JUDGE_ERROR,
            "reason": f"unparseable judge reply: {(text or '')[:200]}",
            "extra_claims": [],
        }
    label = str(raw.get("label", "")).strip().casefold()
    claims = raw.get("extra_claims") or []
    if not isinstance(claims, list):
        claims = [str(claims)]
    if label not in LABELS:
        return {
            "label": JUDGE_ERROR,
            "reason": f"unknown label {label!r}: {raw.get('reason', '')}",
            "extra_claims": [str(c) for c in claims],
        }
    return {
        "label": label,
        "reason": str(raw.get("reason", "")),
        "extra_claims": [str(c) for c in claims],
    }


def _sources_block(passages: list[str]) -> str:
    """Same shape as `guardrail._sources_block`, so the CSV shows what the guardrail saw."""
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))


def format_claims(claims: list[dict]) -> str:
    """One string per verdict for the CSV: `severity: "span" => "replacement" (reason)`,
    joined with ` | `; the `=>` part only when the judge offered a replacement."""
    out = []
    for c in claims:
        item = f'{c.get("severity", "")}: "{c.get("span", "")}"'
        if c.get("replacement"):
            item += f' => "{c["replacement"]}"'
        out.append(f'{item} ({c.get("reason", "")})')
    return " | ".join(out)


REPAIR_PATHS = ("python", "llm", "escalate")


def repair_path(verdict: dict, answer: str) -> str:
    """Which rung of the coordinator's repair ladder this verdict would reach.

    Computed with the same pure functions `Coordinator._repair_or_escalate` uses, so the
    column costs nothing and cannot drift from the real decision:

      ""        the guardrail passed; nothing to repair
      python    every claim is minor and `apply_claims` deletes them safely
      llm       every claim is minor but Python refused -- the repair call would run
      escalate  a critical claim, or no named claim at all

    `llm` is an upper bound: the real path judges the repaired reply once more and may
    still escalate. That second outcome needs a paid call, so it is not simulated here.
    """
    if verdict.get("pass", True):
        return ""
    claims = verdict.get("unsupported_claims") or []
    if not only_minor(claims):
        return "escalate"
    return "python" if apply_claims(answer, claims) is not None else "llm"


_NO_REPAIR = {
    "repair_path": "",
    "repaired_answer": "",
    "repaired_guardrail_pass": "",
    "repaired_unsupported_claims": "",
    "final_escalated": False,
}


async def repair_outcome(
    verdict: dict, answer: str, source_passages: list[str]
) -> tuple[dict, TurnCost]:
    """Run the rung `repair_path` names and judge what it produced.

    Columns:
      repaired_answer             the text after the Python delete or the LLM repair
      repaired_guardrail_pass     the guardrail's verdict on that text ("" when there is none)
      repaired_unsupported_claims that verdict's claims, formatted like `unsupported_claims`
      final_escalated             what production would do with this row

    The Python rung is judged here although production ships it unjudged: the harness
    exists to check the guardrail, and the only way to check the trust-the-delete decision
    is to see how often the judge would still flag a stripped answer. `final_escalated`
    keeps production's semantics regardless -- a Python delete never escalates, an LLM
    repair escalates when the recheck fails or the model returned nothing.

    Costs the LLM repair plus one guardrail call for any repaired row; the escalate rung
    costs nothing.
    """
    path = repair_path(verdict, answer)
    cols = {**_NO_REPAIR, "repair_path": path, "final_escalated": path != ""}
    claims = verdict.get("unsupported_claims") or []
    started = time.perf_counter()
    with usage.collect() as u:
        if path == "python":
            repaired = apply_claims(answer, claims)
        elif path == "llm":
            repaired = await KnowledgeAgent().repair(answer, claims, source_passages)
        else:
            repaired = None
        if repaired is not None:
            recheck = await GuardrailAgent().check_output(repaired, source_passages)
            cols["repaired_answer"] = repaired
            cols["repaired_guardrail_pass"] = bool(recheck.get("pass", True))
            cols["repaired_unsupported_claims"] = format_claims(
                recheck.get("unsupported_claims") or []
            )
            cols["final_escalated"] = path == "llm" and not cols["repaired_guardrail_pass"]
    return cols, _turn_cost(u, started)


def _judge_label(test: TestQuestion, run: AnswerRun) -> tuple[dict, TurnCost]:
    """Label one answer (synchronous - the LLM facade is sync)."""
    content = f"""Question:
{test.question}

SOURCE (passages the answer cited; the QUY TRÌNH above is the other source):
{_sources_block(run.source_passages) or "(none)"}

REFERENCE answer:
{test.reference_answer}

GENERATED answer:
{run.answer}

Return the JSON object described in the instructions."""
    started = time.perf_counter()
    with usage.collect() as u:
        text = complete_text(
            messages=[{"role": "user", "content": content}],
            # Same prefix shape as `GuardrailAgent.check_output`: the process rules are a
            # source for the agent, so they must be one for the judge too.
            system=[PROCESS_BLOCK, {"type": "text", "text": _LABEL_JUDGE_SYSTEM}],
            model=judge_model(),
        )
    return parse_label_reply(text), _turn_cost(u, started)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

# Copied from `run_eval.py` rather than imported: one persistent event loop and one
# RagClient per worker *process*, because the Google embedding client and
# AsyncQdrantClient bind to the first loop that touches them (see the comment there).
# Threads + `asyncio.run` per row, as `triage_eval.py` does, would break on that.

_LOOP: asyncio.AbstractEventLoop | None = None
_RAG: RagClient | None = None


def _rag() -> RagClient:
    global _RAG
    if _RAG is None:
        _RAG = RagClient()
    return _RAG


async def _one(test: TestQuestion, scope: bool) -> dict:
    """KnowledgeAgent -> guardrail -> judge, for one question, on this worker's loop."""
    run = await answer_question(test, _rag(), scope)

    # The same call, with the same two arguments, that `Coordinator.handle_turn` makes.
    # Always made -- the no-passages short-circuit is part of the behaviour under test.
    started = time.perf_counter()
    with usage.collect() as gu:
        verdict = await GuardrailAgent().check_output(run.answer, run.source_passages)
    guard_cost = _turn_cost(gu, started)

    # What the coordinator's repair ladder would do next, and how the result judges.
    repair_cols, repair_cost = await repair_outcome(verdict, run.answer, run.source_passages)

    # Only an answer can be grounded or not. Clarify / no-answer replies are canned text
    # and a suspected-bug reply is a diagnosis, not an answer; they are counted, not rated.
    if run.outcome == "answered":
        judged, judge_cost = await asyncio.to_thread(_judge_label, test, run)
    else:
        judged, judge_cost = {"label": "", "reason": "", "extra_claims": []}, TurnCost()

    return {
        "id": test.id,
        "question": test.question,
        "application": test.application,
        "reference_answer": test.reference_answer,
        "outcome": run.outcome,
        "n_sources": len(run.source_passages),
        "source": _sources_block(run.source_passages),
        "answer": run.answer,
        "answer_citations": "|".join(format_citation(c) for c in run.citations),
        "guardrail_pass": bool(verdict.get("pass", True)),
        "guardrail_reason": verdict.get("reason", ""),
        "unsupported_claims": format_claims(verdict.get("unsupported_claims") or []),
        **repair_cols,
        "judge_label": judged["label"],
        "judge_reason": judged["reason"],
        "judge_extra_claims": " | ".join(judged["extra_claims"]),
        # Empty on purpose: the human override column. `--score` reads it back.
        "label": "",
        "latency_s": round(run.cost.latency_s, 3),
        "cost_usd": run.cost.cost_usd,
        "guardrail_cost_usd": guard_cost.cost_usd,
        "judge_cost_usd": judge_cost.cost_usd,
        "repair_cost_usd": repair_cost.cost_usd,
        "models_used": json.dumps(
            {
                "agent": run.cost.models,
                "guardrail": guard_cost.models,
                "judge": judge_cost.models,
                "repair": repair_cost.models,
            },
            ensure_ascii=False,
        ),
        "error": "",
    }


def _init_worker() -> None:
    global _LOOP
    _LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_LOOP)


def _worker(args: tuple[TestQuestion, bool]) -> dict:
    test, scope = args
    if _LOOP is None:
        _init_worker()
    assert _LOOP is not None
    try:
        return _LOOP.run_until_complete(_one(test, scope))
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
        return {"id": test.id, "question": test.question, "error": f"{type(exc).__name__}: {exc}"}


def run_all(tests: list[TestQuestion], scope: bool, concurrency: int) -> list[dict]:
    """Evaluate every question, preserving input order."""
    total = len(tests)
    started = time.monotonic()
    payload = [(t, scope) for t in tests]
    rows: list[dict] = []

    def note(done: int, row: dict) -> None:
        elapsed = time.monotonic() - started
        eta = (elapsed / done) * (total - done)
        flag = "ERR " if row.get("error") else ""
        print(f"  [{done:3d}/{total}] {flag}{row['id']}  eta {eta:5.0f}s", file=sys.stderr)

    if concurrency <= 1:
        _init_worker()
        for i, item in enumerate(payload, start=1):
            rows.append(_worker(item))
            note(i, rows[-1])
        return rows

    with ProcessPoolExecutor(max_workers=concurrency, initializer=_init_worker) as pool:
        for done, row in enumerate(pool.map(_worker, payload), start=1):
            rows.append(row)
            note(done, row)
    return rows


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def scored(rows: list[dict]) -> list[dict]:
    """Rows that ran. Errored rows are counted, never averaged."""
    return [r for r in rows if not r.get("error")]


def rated(rows: list[dict]) -> list[dict]:
    """Rows the rate is computed over: an actual answer with a usable label."""
    return [r for r in scored(rows) if r.get("outcome") == "answered" and final_label(r) in LABELS]


def false_escalate_rate(rows: list[dict]) -> tuple[float, int, int]:
    """`(rate, escalated, total)` over grounded + minor answers.

    Critical answers are excluded from the denominator: escalating one is the guardrail
    doing its job, and letting one through is the other metric, which is not tracked here.
    """
    good = [r for r in rated(rows) if final_label(r) in ("grounded", "minor")]
    escalated = sum(1 for r in good if not r["guardrail_pass"])
    return (escalated / len(good) if good else 0.0), escalated, len(good)


def false_escalate_rate_after_repair(rows: list[dict]) -> tuple[float, int, int]:
    """Same denominator as `false_escalate_rate`, counting only rows production would
    still hand off once the repair ladder has run (`final_escalated`)."""
    good = [r for r in rated(rows) if final_label(r) in ("grounded", "minor")]
    escalated = sum(1 for r in good if r.get("final_escalated") is True)
    return (escalated / len(good) if good else 0.0), escalated, len(good)


def label_matrix(rows: list[dict]) -> dict[str, dict[str, int]]:
    """`label -> {pass, escalate}` over rated rows."""
    matrix = {label: {"pass": 0, "escalate": 0} for label in LABELS}
    for r in rated(rows):
        matrix[final_label(r)]["pass" if r["guardrail_pass"] else "escalate"] += 1
    return matrix


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Small n makes interpolation meaningless here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[idx]


def _sum_cost(rows: list[dict], column: str) -> float | None:
    """Total of one cost column; `None` when nothing priced (never a misleading $0)."""
    priced = [float(r[column]) for r in rows if r.get(column) not in (None, "")]
    return sum(priced) if priced else None


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _clip(text: str, width: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _matrix_block(rows: list[dict]) -> list[str]:
    matrix = label_matrix(rows)
    out = ["  " + "label".ljust(12) + "   pass  escalate"]
    for label in LABELS:
        m = matrix[label]
        out.append("  " + label.ljust(12) + f"{m['pass']:7d}{m['escalate']:10d}")
    return out


def _false_escalation_block(rows: list[dict]) -> list[str]:
    wrong = [r for r in rated(rows) if final_label(r) != "critical" and not r["guardrail_pass"]]
    if not wrong:
        return ["FALSE ESCALATIONS (0)", "  none — every grounded/minor answer passed"]
    out = [f"FALSE ESCALATIONS ({len(wrong)})"]
    for r in wrong:
        out.append(
            f"  {r['id']}  {final_label(r)}  repair={r.get('repair_path') or 'escalate'}  "
            f"guardrail: {_clip(r['guardrail_reason'], 90)}"
        )
        if r.get("unsupported_claims"):
            out.append(f"        unsupported: {_clip(r['unsupported_claims'], 90)}")
        if r.get("repaired_answer"):
            out.append(
                f"        repaired (pass={r.get('repaired_guardrail_pass')}): "
                f"{_clip(r['repaired_answer'], 200)}"
            )
        if r.get("judge_reason"):
            out.append(f"        judge: {_clip(r['judge_reason'], 90)}")
        out.append(f"        answer: {_clip(r['answer'], 200)}")
    return out


def _critical_block(rows: list[dict]) -> list[str]:
    crit = [r for r in rated(rows) if final_label(r) == "critical"]
    if not crit:
        return ["CRITICAL ANSWERS (0)", "  none"]
    out = [f"CRITICAL ANSWERS ({len(crit)})"]
    for r in crit:
        out.append(
            f"  {r['id']}  pass={r['guardrail_pass']}  judge: {_clip(r.get('judge_reason', ''), 90)}"
        )
    return out


def summarise(rows: list[dict], tests_file: str = TEST_FILE, wall_s: float = 0.0) -> str:
    ok = scored(rows)
    errors = len(rows) - len(ok)
    outcomes = {
        o: sum(1 for r in ok if r.get("outcome") == o)
        for o in ("answered", "clarify", "no_answer", "suspected_bug")
    }
    answered = [r for r in ok if r.get("outcome") == "answered"]
    counts = {label: sum(1 for r in answered if final_label(r) == label) for label in LABELS}
    judge_errors = sum(1 for r in answered if final_label(r) == JUDGE_ERROR)
    human = sum(1 for r in answered if r.get("label"))
    rate, k, n = false_escalate_rate(rows)
    crit_passed = sum(
        1 for r in rated(rows) if final_label(r) == "critical" and r["guardrail_pass"]
    )
    # Over every failed row, rated or not: this is about what the ladder would do with
    # the guardrail's verdicts, not about whether the judge agreed with them.
    failed = [r for r in ok if r.get("guardrail_pass") is False]
    paths = {
        p: sum(1 for r in failed if (r.get("repair_path") or "escalate") == p)
        for p in REPAIR_PATHS
    }
    after_rate, after_k, after_n = false_escalate_rate_after_repair(rows)
    python_rows = [r for r in failed if r.get("repair_path") == "python"]
    python_flagged = sum(1 for r in python_rows if r.get("repaired_guardrail_pass") is False)
    latencies = [float(r["latency_s"]) for r in ok if r.get("latency_s") not in (None, "")]

    cfg = get_settings()
    lines = [
        "",
        f"GROUNDING GUARDRAIL ({len(rows)} rows, tests={Path(tests_file).name},",
        f"  knowledge={cfg.model_for('knowledge')}, guardrail={cfg.model_for('guardrail')}, "
        f"judge={judge_model()})",
        f"  answered   {outcomes['answered']:5d}   clarify {outcomes['clarify']}  "
        f"no_answer {outcomes['no_answer']}  suspected_bug {outcomes['suspected_bug']}  "
        f"errors {errors}",
        f"  labels     grounded {counts['grounded']}   minor {counts['minor']}   "
        f"critical {counts['critical']}   judge_error {judge_errors}   "
        f"(human overrides: {human})",
        "",
        *_matrix_block(rows),
        "",
        f"  FALSE ESCALATE RATE   {100 * rate:6.2f}%   ({k} / {n} grounded+minor answers escalated)",
        f"  AFTER REPAIR          {100 * after_rate:6.2f}%   ({after_k} / {after_n} still escalated "
        "once the repair ladder has run)",
        f"  critical passed       {crit_passed:6d}    (informational; missed error rate not tracked)",
        f"  repair path           python {paths['python']}   llm {paths['llm']}   "
        f"escalate {paths['escalate']}   (over {len(failed)} guardrail failures)",
        f"  python deletes the judge still flags   {python_flagged} / {len(python_rows)}"
        "   (informational; production ships them unjudged)",
        "",
        f"  cost   agent {fmt_usd(_sum_cost(ok, 'cost_usd'))}   "
        f"guardrail {fmt_usd(_sum_cost(ok, 'guardrail_cost_usd'))}   "
        f"repair {fmt_usd(_sum_cost(ok, 'repair_cost_usd'))}   "
        f"judge {fmt_usd(_sum_cost(ok, 'judge_cost_usd'))}",
        f"  agent latency mean/p50/p95   "
        f"{(sum(latencies) / len(latencies) if latencies else 0.0):6.2f}s"
        f"{_percentile(latencies, 50):7.2f}s{_percentile(latencies, 95):7.2f}s"
        f"      wall {wall_s:.1f}s",
        "",
        *_false_escalation_block(rows),
        "",
        *_critical_block(rows),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CSV in / out
# --------------------------------------------------------------------------


def write_rows(rows: list[dict], path: Path) -> None:
    """Per-row CSV. Copied from `triage_eval.write_rows` for the same reason it copied it."""
    fields: list[str] = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return (value or "").strip().casefold() in {"true", "yes", "1"}


def _as_tri_bool(value: str | bool) -> bool | str:
    """A bool column that may also be unset: `""` stays `""`."""
    if isinstance(value, bool) or not (value or "").strip():
        return value if isinstance(value, bool) else ""
    return _as_bool(value)


def load_scored(path: str | Path) -> list[dict]:
    """Re-read a results CSV, typically after a human filled in `label`.

    CSV gives strings back for everything, so the columns the metrics branch on are
    restored to their types here; the rest is only printed. `utf-8-sig` because the file
    has usually been through Excel by the time it is re-scored.
    """
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            row_id = row.get("id", "?")
            row["guardrail_pass"] = _as_bool(row.get("guardrail_pass", ""))
            row["final_escalated"] = _as_bool(row.get("final_escalated", ""))
            row["repaired_guardrail_pass"] = _as_tri_bool(row.get("repaired_guardrail_pass", ""))
            row["label"] = normalise_label(row.get("label"), row_id)
            row["judge_label"] = (row.get("judge_label") or "").strip()
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run KnowledgeAgent + output guardrail, label answers, report false escalations"
    )
    ap.add_argument(
        "--tests", default=TEST_FILE, help=f"test-set CSV (default {Path(TEST_FILE).name})"
    )
    ap.add_argument("--limit", type=int, help="evaluate only the first N rows")
    ap.add_argument("--application", help="restrict to one application (slug or display name)")
    ap.add_argument(
        "--no-scope", action="store_true", help="retrieve without the application filter"
    )
    ap.add_argument("--concurrency", type=int, default=6, help="worker processes")
    ap.add_argument("--out", help="per-row CSV path (default eval/results/guardrail-<ts>.csv)")
    ap.add_argument(
        "--model",
        help="run the whole pipeline on this model (sets every per-agent model var)",
    )
    ap.add_argument(
        "--guardrail-model",
        help="guardrail model only (sets GUARDRAIL_MODEL) -- the knob this harness exists to tune",
    )
    ap.add_argument("--judge-model", help="label judge model (sets EVAL_JUDGE_MODEL)")
    ap.add_argument(
        "--score",
        metavar="CSV",
        help="re-score an existing results CSV (after filling `label`); no LLM or Qdrant calls",
    )
    args = ap.parse_args()

    if args.score:
        try:
            rows = load_scored(args.score)
        except (OSError, ValueError) as exc:
            print(f"Could not load {args.score}: {exc}", file=sys.stderr)
            sys.exit(1)
        report = summarise(rows, tests_file=args.tests)
        print(report)
        Path(args.score).with_suffix(".txt").write_text(report, encoding="utf-8")
        return

    # Set before the worker pool exists so the children inherit them; `get_settings` is
    # lru_cached and may already be warm in this process.
    if args.model:
        for var in _MODEL_ENV_VARS:
            os.environ[var] = args.model
    if args.guardrail_model:
        os.environ["GUARDRAIL_MODEL"] = args.guardrail_model
    if args.judge_model:
        os.environ["EVAL_JUDGE_MODEL"] = args.judge_model
    if args.model or args.guardrail_model or args.judge_model:
        get_settings.cache_clear()

    tests = load_tests(path=args.tests, application=args.application)
    if not tests:
        print(f"No rows in {args.tests} (application={args.application!r})", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        tests = tests[: args.limit]
    scope = not args.no_scope

    cfg = get_settings()
    print(
        f"Running {len(tests)} rows (knowledge={cfg.model_for('knowledge')}, "
        f"guardrail={cfg.model_for('guardrail')}, judge={judge_model()}, scoped={scope}, "
        f"concurrency={args.concurrency}, tests={Path(args.tests).name})",
        file=sys.stderr,
    )
    started = time.monotonic()
    rows = run_all(tests, scope, args.concurrency)
    wall_s = time.monotonic() - started

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else RESULTS_DIR / f"guardrail-{stamp}.csv"
    write_rows(rows, out)

    report = summarise(rows, tests_file=args.tests, wall_s=wall_s)
    print(report)
    print(f"\nPer-row results: {out}")
    print("Fill the `label` column (grounded | minor | critical) to override the judge, then:")
    print(f"  poetry run python -m eval.guardrail_eval --score {out}")
    out.with_suffix(".txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
