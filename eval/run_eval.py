"""Batch-run the golden test set and report aggregate retrieval + answer metrics.

    poetry run python -m eval.run_eval                      # everything
    poetry run python -m eval.run_eval --mode retrieval     # no LLM cost
    poetry run python -m eval.run_eval --limit 10           # smoke test
    poetry run python -m eval.run_eval --no-scope           # A/B the application filter
    poetry run python -m eval.run_eval --mode answer \
        --model gpt-5.4-mini --judge-model gpt-5.4-mini     # A/B a model on quality + cost

Per-row results land in `eval/results/<mode>-<timestamp>.csv`; the summary prints
to stdout.

Scoring rules worth knowing before reading the numbers:

  * Retrieval metrics are averaged over the 91 *answerable* rows only. The 9
    adversarial rows have no correct passage to retrieve -- their keywords are
    phrases like "không có thông tin" -- so including them would depress MRR for
    behaving correctly.
  * The 6 off-topic rows expect the triage out_of_scope route to block them.
    This runner exercises KnowledgeAgent directly and does not run triage, and
    KnowledgeAgent deliberately carries no scope logic — so those rows will show
    answered/clarify here; only the full Coordinator path refuses them.
  * Cost and latency cover *all* rows that ran, adversarial ones included -- a
    refusal costs money too. Agent spend and judge spend are reported separately:
    the judge is pinned and runs on every row whatever model is under test, so
    merging them would add a constant that hides the difference being measured.
    Prices come from `eval/pricing.json`; a model missing from it is reported as
    unpriced and excluded, never billed at zero.
"""

import argparse
import asyncio
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from agent_customer_support.config import get_settings
from agent_customer_support.rag_client import RagClient
from eval.eval import (
    DEFAULT_K,
    AnswerEval,
    AnswerRun,
    RetrievalEval,
    evaluate_answer,
    evaluate_retrieval,
    fmt_usd,
    judge_model,
)
from eval.pricing import cost_by_model
from eval.testset import TEST_FILE, TestQuestion, load_tests

RESULTS_DIR = Path(__file__).parent / "results"

# Every model knob in Settings. `--model` sets all of them, so a run is one model
# end to end -- otherwise the cost number mixes the answer model with whatever
# `KNOWLEDGE_CONTEXTUALIZE_MODEL` happens to be, and two runs are not comparable.
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
# Execution
# --------------------------------------------------------------------------


# --- worker-process state ------------------------------------------------
#
# Parallelism here has one hard constraint: a given *process* must use exactly one
# event loop for its whole lifetime. The Google embedding client is an `lru_cache`d
# singleton (`rag/embeddings._client`) whose async transport binds to the first loop
# that touches it, and `AsyncQdrantClient` behaves the same way. That rules out
# threads (they would share that one cached client across loops -- "Future attached
# to a different loop") and rules out `asyncio.run` per question (a fresh loop each
# call). Hence worker processes, each holding a persistent loop and one RagClient.
#
# A single shared loop would also be correct, but it serialises the agent's compose
# step: that is a blocking sync LLM call made from inside KnowledgeAgent, where the
# harness has no seam to offload it to a thread.

_LOOP: asyncio.AbstractEventLoop | None = None
_RAG: RagClient | None = None


def _rag() -> RagClient:
    """One RagClient per worker process, bound to that worker's loop."""
    global _RAG
    if _RAG is None:
        _RAG = RagClient()
    return _RAG


async def _one(test: TestQuestion, mode: str, k: int, scope: bool) -> dict:
    """Evaluate a single question on the calling worker's event loop."""
    rag = _rag()
    row: dict = {
        "id": test.id,
        "question": test.question,
        "application": test.application,
        "category": test.category,
        "difficulty": test.difficulty,
        "answerable": test.answerable,
        "multi_hop": test.multi_hop,
        "expected_route": test.expected_route,
    }
    if mode in ("retrieval", "both"):
        r: RetrievalEval = await evaluate_retrieval(test, k, scope, rag)
        row |= {
            "mrr": round(r.mrr, 4),
            "ndcg": round(r.ndcg, 4),
            "keyword_coverage": round(r.keyword_coverage, 1),
            "keywords_found": r.keywords_found,
            "total_keywords": r.total_keywords,
            "hit": r.hit,
            "n_passages": r.n_passages,
            "top_confidence": round(r.top_confidence, 4),
            "scope_precision": round(r.scope_precision, 3),
            "source_doc_hit": r.source_doc_hit,
            "keywords": " | ".join(test.keywords),
            "retrieved": " ".join(r.hit_labels),
            # Readable sources; `citation_ids` keeps the UUIDs the agent reports.
            "citations": " ; ".join(r.hit_sources),
            "citation_ids": "|".join(r.citation_ids),
            "snippets": "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(r.snippets)),
            # "keyword@rank", rank 0 meaning it appeared in no retrieved passage.
            "keyword_ranks": " | ".join(f"{kw}@{rk}" for kw, rk in r.keyword_ranks.items()),
            "missed_keywords": " | ".join(kw for kw, rk in r.keyword_ranks.items() if rk == 0),
        }
    if mode in ("answer", "both"):
        verdict: AnswerEval
        run: AnswerRun
        verdict, run = await evaluate_answer(test, scope, rag)
        row |= {
            "outcome": run.outcome,
            "abstained": run.abstained,
            "accuracy": verdict.accuracy,
            "completeness": verdict.completeness,
            "relevance": verdict.relevance,
            "faithfulness": verdict.faithfulness,
            "feedback": verdict.feedback,
            "answer": run.answer,
            # Distinct from the retrieval block's `citations`: this is what the agent
            # attached to its reply, which also carries `qa:`-prefixed Q&A hits.
            "answer_citations": "|".join(run.citations),
            # Cost/latency of the agent under test, kept apart from the judge's own
            # spend so a model comparison is not diluted by the fixed judge.
            "latency_s": round(run.cost.latency_s, 3),
            "llm_calls": run.cost.n_calls,
            "input_tokens": run.cost.input_tokens,
            "output_tokens": run.cost.output_tokens,
            "cost_usd": run.cost.cost_usd,
            "models_used": json.dumps(run.cost.models, ensure_ascii=False),
            "judge_latency_s": round(verdict.cost.latency_s, 3),
            "judge_input_tokens": verdict.cost.input_tokens,
            "judge_output_tokens": verdict.cost.output_tokens,
            "judge_cost_usd": verdict.cost.cost_usd,
            "judge_models_used": json.dumps(verdict.cost.models, ensure_ascii=False),
        }
    return row


def _init_worker() -> None:
    """Give this worker process one event loop, for its whole lifetime."""
    global _LOOP
    _LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_LOOP)


def _worker(args: tuple[TestQuestion, str, int, bool]) -> dict:
    """Worker entry point: run one question on this process's persistent loop."""
    test, mode, k, scope = args
    if _LOOP is None:
        _init_worker()
    assert _LOOP is not None
    try:
        return _LOOP.run_until_complete(_one(test, mode, k, scope))
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
        return {"id": test.id, "question": test.question, "error": f"{type(exc).__name__}: {exc}"}


def run_all(
    tests: list[TestQuestion], mode: str, k: int, scope: bool, concurrency: int
) -> list[dict]:
    total = len(tests)
    started = time.monotonic()
    payload = [(t, mode, k, scope) for t in tests]
    rows: list[dict] = []

    def note(done: int, row: dict) -> None:
        elapsed = time.monotonic() - started
        eta = (elapsed / done) * (total - done)
        flag = "ERR " if "error" in row else ""
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
# Aggregation
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _pct(flags: list[bool]) -> float:
    return 100.0 * sum(1 for f in flags if f) / len(flags) if flags else 0.0


def _retrieval_block(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "MRR": _mean([r["mrr"] for r in rows]),
        "nDCG": _mean([r["ndcg"] for r in rows]),
        "KeywordCov%": _mean([r["keyword_coverage"] for r in rows]),
        "Hit%": _pct([r["hit"] for r in rows]),
        "SrcDoc%": _pct([r["source_doc_hit"] for r in rows]),
        "Scope": _mean([r["scope_precision"] for r in rows]),
        "Passages": _mean([float(r["n_passages"]) for r in rows]),
        "Empty%": _pct([r["n_passages"] == 0 for r in rows]),
    }


def _clip(text: str, width: int) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def retrieval_detail(rows: list[dict], full: bool = False) -> str:
    """Per-question retrieval table: question, keywords, citations, MRR, nDCG, coverage, hit.

    The aggregate table says an application retrieves badly; this one says which
    questions and which keywords. `full` adds a per-question breakdown with the rank
    each keyword landed at and the chunks that came back, which is what actually
    identifies whether the right chunk was missed or merely ranked low.
    """
    if not rows:
        return ""
    qw, kw, cw = 46, 34, 30
    head = (
        f"{'ID':<6}{'QUESTION':<{qw + 2}}{'KEYWORDS':<{kw + 2}}{'RETRIEVED':<{cw + 2}}"
        f"{'MRR':>7}{'nDCG':>7}{'Cov%':>7}{'Hit':>5}"
    )
    lines = ["\nPER-QUESTION RETRIEVAL", "-" * len(head), head, "-" * len(head)]
    for r in sorted(rows, key=lambda x: x["mrr"]):
        lines.append(
            f"{r['id']:<6}{_clip(r['question'], qw):<{qw + 2}}"
            f"{_clip(r.get('keywords', ''), kw):<{kw + 2}}"
            f"{_clip(r.get('retrieved', ''), cw):<{cw + 2}}"
            f"{r['mrr']:>7.3f}{r['ndcg']:>7.3f}{r['keyword_coverage']:>7.1f}"
            f"{('yes' if r['hit'] else 'NO'):>5}"
        )
    if not full:
        return "\n".join(lines)

    lines.append("\nPER-QUESTION BREAKDOWN  (keyword@rank, rank 0 = in none of the top-k)")
    lines.append("-" * len(head))
    for r in sorted(rows, key=lambda x: x["mrr"]):
        lines.append(f"\n{r['id']}  {r['question']}")
        lines.append(
            f"    mrr={r['mrr']:.3f}  ndcg={r['ndcg']:.3f}  cov={r['keyword_coverage']:.0f}%  "
            f"passages={r['n_passages']}  top_conf={r['top_confidence']:.3f}  "
            f"src_doc_hit={r['source_doc_hit']}"
        )
        lines.append(f"    keywords : {r.get('keyword_ranks', '')}")
        if r.get("missed_keywords"):
            lines.append(f"    MISSED   : {r['missed_keywords']}")
        lines.append("    retrieved passages (score, source file, chunk):")
        sources = [s for s in (r.get("citations") or "").split(" ; ") if s]
        snippets = (r.get("snippets") or "").split("\n")
        for i, src in enumerate(sources):
            lines.append(f"      {i + 1}. {src}")
            if i < len(snippets) and snippets[i]:
                # Strip the "[n] " prefix added when the snippets column was built.
                lines.append(f"         {snippets[i].split('] ', 1)[-1]}")
    return "\n".join(lines)


def _answer_block(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "Accuracy": _mean([r["accuracy"] for r in rows]),
        "Complete": _mean([r["completeness"] for r in rows]),
        "Relevance": _mean([r["relevance"] for r in rows]),
        "Faithful": _mean([r["faithfulness"] for r in rows]),
        "Answered%": _pct([not r["abstained"] for r in rows]),
    }


def _merge_models(rows: list[dict], column: str) -> dict[str, list[int]]:
    """Sum the per-row `model -> [in, out, calls]` maps into one."""
    totals: dict[str, list[int]] = {}
    for r in rows:
        for model, counts in json.loads(r.get(column) or "{}").items():
            entry = totals.setdefault(model, [0, 0, 0])
            for i in range(3):
                entry[i] += counts[i]
    return totals


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: the smallest value at or above `pct` of the sample.

    No interpolation -- a latency percentile should be a latency that actually
    happened. Rank is `ceil(pct/100 * n)`, 1-based, so p50 of [3, 9] is 3 and p95 is 9.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def cost_section(rows: list[dict], wall_s: float, concurrency: int) -> tuple[str, list[str]]:
    """COST & LATENCY block, plus the models that burned tokens with no price entry.

    Cost is recomputed from the *summed* token counts rather than by adding the
    per-row costs: an unpriced model yields None per row, and summing Nones would
    either crash or quietly read as zero. Summing tokens first keeps the priced part
    exact and the unpriced part visible.
    """
    if not rows:
        return "", []

    agent_models = _merge_models(rows, "models_used")
    judge_models = _merge_models(rows, "judge_models_used")
    agent_cost, agent_unpriced = cost_by_model(agent_models)
    judge_cost, judge_unpriced = cost_by_model(judge_models)

    n = len(rows)
    tokens_in = sum(int(r.get("input_tokens") or 0) for r in rows)
    tokens_out = sum(int(r.get("output_tokens") or 0) for r in rows)
    calls = sum(int(r.get("llm_calls") or 0) for r in rows)
    latencies = [float(r.get("latency_s") or 0.0) for r in rows]

    total = (
        None if agent_cost is None and judge_cost is None else (agent_cost or 0) + (judge_cost or 0)
    )
    per_row = None if agent_cost is None else agent_cost / n

    out = ["\nCOST & LATENCY (all evaluated rows)", "-" * 64]
    out.append(f"  rows                      {n:>10d}")
    out.append(
        f"  agent cost                {fmt_usd(agent_cost):>10}"
        f"     judge {fmt_usd(judge_cost)}     total {fmt_usd(total)}"
    )
    out.append(f"  agent cost / row          {fmt_usd(per_row, 6):>10}")
    if per_row:
        out.append(f"  agent cost / 1000 rows    {fmt_usd(per_row * 1000, 2):>10}")
    out.append(
        f"  tokens in / out           {tokens_in:>10,} / {tokens_out:,}"
        f"     ({tokens_in / n:,.0f} / {tokens_out / n:,.0f} per row)"
    )
    out.append(f"  llm calls / row           {calls / n:>10.2f}")
    out.append(
        f"  latency mean/p50/p95      {_mean(latencies):>10.2f}s"
        f"    {_percentile(latencies, 50):.2f}s  {_percentile(latencies, 95):.2f}s"
    )
    out.append(f"  wall clock                {wall_s:>10.0f}s     (concurrency {concurrency})")

    out.append("\n  by model (input / output tokens, calls, cost)")
    for label, models in (("agent", agent_models), ("judge", judge_models)):
        for model, (t_in, t_out, n_calls) in sorted(models.items()):
            one, _ = cost_by_model({model: [t_in, t_out, n_calls]})
            out.append(
                f"    {label:<6}{model:<28}{t_in:>10,} /{t_out:>9,}{n_calls:>7}   {fmt_usd(one)}"
            )

    unpriced = sorted(set(agent_unpriced) | set(judge_unpriced))
    if unpriced:
        out.append("")
        out.append("  !! NO PRICE ENTRY in eval/pricing.json -- cost above EXCLUDES these:")
        for model in unpriced:
            out.append(f"       {model}")
    return "\n".join(out), unpriced


def _table(title: str, blocks: dict[str, dict]) -> str:
    if not blocks:
        return ""
    cols = list(next(iter(blocks.values())).keys())
    label_w = max(len(k) for k in blocks) + 2
    head = f"{'':<{label_w}}" + "".join(f"{c:>12}" for c in cols)
    lines = [f"\n{title}", "-" * len(head), head, "-" * len(head)]
    for label, vals in blocks.items():
        cells = "".join(f"{v:>12d}" if isinstance(v, int) else f"{v:>12.3f}" for v in vals.values())
        lines.append(f"{label:<{label_w}}{cells}")
    return "\n".join(lines)


def summarise(
    rows: list[dict],
    mode: str,
    k: int,
    scope: bool,
    detail: str = "none",
    tests_file: str = "",
    wall_s: float = 0.0,
    concurrency: int = 1,
) -> str:
    ok = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]
    cfg = get_settings()
    out: list[str] = []

    out.append("=" * 80)
    out.append(f"RAG EVALUATION  |  mode={mode}  top_k={k}  scoped={scope}")
    out.append(
        f"agent={cfg.model_for('knowledge')}  contextualize="
        f"{cfg.model_for('knowledge_contextualize')}  judge={judge_model()}"
    )
    out.append(f"tests={tests_file or TEST_FILE}")
    out.append(f"rows={len(rows)}  ok={len(ok)}  errors={len(errors)}")
    out.append("=" * 80)

    answerable = [r for r in ok if r["answerable"]]
    adversarial = [r for r in ok if not r["answerable"]]

    if mode in ("retrieval", "both") and answerable:
        out.append(
            _table("RETRIEVAL (answerable rows only)", {"overall": _retrieval_block(answerable)})
        )

        by_app: dict[str, list[dict]] = defaultdict(list)
        for r in answerable:
            by_app[r["application"] or "(none)"].append(r)
        out.append(
            _table(
                "RETRIEVAL by application",
                {a: _retrieval_block(v) for a, v in sorted(by_app.items())},
            )
        )

        by_diff: dict[str, list[dict]] = defaultdict(list)
        for r in answerable:
            by_diff[r["difficulty"]].append(r)
        order = ["easy", "medium", "hard"]
        out.append(
            _table(
                "RETRIEVAL by difficulty",
                {d: _retrieval_block(by_diff[d]) for d in order if by_diff.get(d)},
            )
        )

        multi = [r for r in answerable if r["multi_hop"]]
        if multi:
            out.append(_table("RETRIEVAL multi-hop", {"multi_hop": _retrieval_block(multi)}))

        if detail != "none":
            out.append(retrieval_detail(answerable, full=(detail == "full")))

    if mode in ("answer", "both") and answerable:
        out.append(
            _table("ANSWER QUALITY (answerable rows)", {"overall": _answer_block(answerable)})
        )

        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in answerable:
            by_cat[r["category"]].append(r)
        out.append(
            _table(
                "ANSWER QUALITY by category",
                {c: _answer_block(v) for c, v in sorted(by_cat.items())},
            )
        )

        outcomes: dict[str, int] = defaultdict(int)
        for r in answerable:
            outcomes[r["outcome"]] += 1
        out.append("\nOUTCOMES (answerable rows)")
        out.append("-" * 40)
        for name, count in sorted(outcomes.items(), key=lambda kv: -kv[1]):
            out.append(f"  {name:<16} {count:3d}  ({100 * count / len(answerable):.0f}%)")

    if mode in ("answer", "both") and adversarial:
        # These rows are correct when the agent declines. Accuracy is still scored
        # (the judge is told the ideal answer admits ignorance) and faithfulness <= 2
        # is the hallucination signal.
        out.append("\nADVERSARIAL ROWS (should not be answered)")
        out.append("-" * 40)
        out.append(f"  rows                 {len(adversarial):3d}")
        out.append(f"  abstained            {_pct([r['abstained'] for r in adversarial]):.0f}%")
        out.append(f"  mean accuracy        {_mean([r['accuracy'] for r in adversarial]):.2f}/5")
        out.append(
            f"  mean faithfulness    {_mean([r['faithfulness'] for r in adversarial]):.2f}/5"
        )
        halluc = [r for r in adversarial if r["faithfulness"] <= 2]
        out.append(f"  hallucinated (F<=2)  {len(halluc):3d}")
        for r in halluc:
            out.append(f"      {r['id']}  {r['question'][:64]}")

    if errors:
        out.append("\nERRORS")
        out.append("-" * 40)
        for r in errors:
            out.append(f"  {r['id']}  {r['error']}")

    # Adversarial rows are billed too -- you pay for a refusal -- so the cost block
    # covers every row that ran, not just the answerable ones.
    if mode in ("answer", "both") and ok:
        block, _unpriced = cost_section(ok, wall_s, concurrency)
        out.append(block)

    if mode in ("answer", "both"):
        weak = sorted(
            (r for r in ok if r["answerable"]),
            key=lambda r: (r["accuracy"], r["faithfulness"]),
        )[:10]
        out.append("\nWEAKEST ANSWERABLE ROWS")
        out.append("-" * 40)
        for r in weak:
            out.append(
                f"  {r['id']}  acc={r['accuracy']:.0f} faith={r['faithfulness']:.0f} "
                f"mrr={r.get('mrr', float('nan')):.2f} [{r['outcome']}] {r['question'][:56]}"
            )

    return "\n".join(out)


# --------------------------------------------------------------------------


def write_rows(rows: list[dict], path: Path) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch RAG evaluation over eval/test.csv")
    ap.add_argument("--mode", choices=["retrieval", "answer", "both"], default="retrieval")
    ap.add_argument("--limit", type=int, help="evaluate only the first N rows")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help=f"top_k (default {DEFAULT_K})")
    ap.add_argument(
        "--no-scope", action="store_true", help="disable the application filter (A/B baseline)"
    )
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", help="per-row CSV path (default eval/results/<mode>-<ts>.csv)")
    ap.add_argument(
        "--application",
        help="restrict to one application (slug or display name), e.g. lay_mau_quan_trac",
    )
    ap.add_argument(
        "--detail",
        choices=["none", "table", "full"],
        default="none",
        help="per-question retrieval output: table of scores, or full keyword/chunk breakdown",
    )
    ap.add_argument(
        "--tests",
        default=TEST_FILE,
        help=f"test-set CSV (default {Path(TEST_FILE).name})",
    )
    ap.add_argument(
        "--model",
        help="run the whole pipeline on this model (overrides AGENT_MODEL and every "
        "per-agent override) so an A/B compares like with like",
    )
    ap.add_argument(
        "--judge-model",
        help="model for the LLM judge (sets EVAL_JUDGE_MODEL); pin it while swapping --model",
    )
    args = ap.parse_args()

    # Set before the worker pool exists: `spawn` and `fork` both hand os.environ to
    # children, and pydantic-settings reads the environment ahead of `.env` (which
    # eval/__init__.py loads with override=False), so these win. `get_settings` is
    # lru_cached, and at --concurrency 1 the parent's cache may already be warm.
    if args.model:
        for var in _MODEL_ENV_VARS:
            os.environ[var] = args.model
    if args.judge_model:
        os.environ["EVAL_JUDGE_MODEL"] = args.judge_model
    if args.model or args.judge_model:
        get_settings.cache_clear()

    tests = load_tests(path=args.tests, application=args.application)
    if not tests:
        print(f"No rows matched --application {args.application!r}", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        tests = tests[: args.limit]
    scope = not args.no_scope

    scope_note = f", application={args.application}" if args.application else ""
    model_note = f", model={args.model}" if args.model else ""
    print(
        f"Running {len(tests)} tests (mode={args.mode}, k={args.k}, scoped={scope}"
        f"{scope_note}{model_note}, tests={Path(args.tests).name})",
        file=sys.stderr,
    )
    started = time.monotonic()
    rows = run_all(tests, args.mode, args.k, scope, args.concurrency)
    wall_s = time.monotonic() - started

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = "" if scope else "-noscope"
    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.mode}{suffix}-{stamp}.csv"
    write_rows(rows, out)

    report = summarise(
        rows,
        args.mode,
        args.k,
        scope,
        args.detail,
        tests_file=args.tests,
        wall_s=wall_s,
        concurrency=args.concurrency,
    )
    print(report)
    print(f"\nPer-row results: {out}")
    out.with_suffix(".txt").write_text(report, encoding="utf-8")

    # Repeat the unpriced warning on stderr: the report is long and this one silently
    # makes the cost number too low, so it must not scroll past unnoticed.
    ok = [r for r in rows if "error" not in r]
    if args.mode in ("answer", "both") and ok:
        _, unpriced = cost_section(ok, wall_s, args.concurrency)
        if unpriced:
            print(
                f"\nWARNING: no price entry in eval/pricing.json for: {', '.join(unpriced)}\n"
                "         Reported cost excludes them. Add them and re-read the report.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
