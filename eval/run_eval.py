"""Batch-run the golden test set and report aggregate retrieval + answer metrics.

    poetry run python -m eval.run_eval                      # everything
    poetry run python -m eval.run_eval --mode retrieval     # no LLM cost
    poetry run python -m eval.run_eval --limit 10           # smoke test
    poetry run python -m eval.run_eval --no-scope           # A/B the application filter

Per-row results land in `eval/results/<mode>-<timestamp>.csv`; the summary prints
to stdout.

Scoring rules worth knowing before reading the numbers:

  * Retrieval metrics are averaged over the 91 *answerable* rows only. The 9
    adversarial rows have no correct passage to retrieve -- their keywords are
    phrases like "không có thông tin" -- so including them would depress MRR for
    behaving correctly.
  * The 3 off-topic rows expect the input guardrail to block them. This runner
    exercises KnowledgeAgent directly and does not run the guardrail, so those
    rows measure the second line of defence: what the RAG layer does if an
    off-topic question reaches it.
"""

import argparse
import asyncio
import csv
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from agent_customer_support.rag_client import RagClient
from eval.eval import (
    DEFAULT_K,
    AnswerEval,
    AnswerRun,
    RetrievalEval,
    evaluate_answer,
    evaluate_retrieval,
    judge_model,
)
from eval.testset import TestQuestion, load_tests

RESULTS_DIR = Path(__file__).parent / "results"


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


def summarise(rows: list[dict], mode: str, k: int, scope: bool, detail: str = "none") -> str:
    ok = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]
    out: list[str] = []

    out.append("=" * 80)
    out.append(f"RAG EVALUATION  |  mode={mode}  top_k={k}  scoped={scope}  judge={judge_model()}")
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
    args = ap.parse_args()

    tests = load_tests(application=args.application)
    if not tests:
        print(f"No rows matched --application {args.application!r}", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        tests = tests[: args.limit]
    scope = not args.no_scope

    scope_note = f", application={args.application}" if args.application else ""
    print(
        f"Running {len(tests)} tests (mode={args.mode}, k={args.k}, scoped={scope}{scope_note})",
        file=sys.stderr,
    )
    rows = run_all(tests, args.mode, args.k, scope, args.concurrency)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = "" if scope else "-noscope"
    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.mode}{suffix}-{stamp}.csv"
    write_rows(rows, out)

    report = summarise(rows, args.mode, args.k, scope, args.detail)
    print(report)
    print(f"\nPer-row results: {out}")
    out.with_suffix(".txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
