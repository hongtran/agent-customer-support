"use client";

import { useEffect, useState } from "react";
import { listQA, createQA, approveQA, rejectQA, editQA } from "@/lib/api";

type Draft = { question: string; answer: string; application: string };

const EMPTY_DRAFT: Draft = { question: "", answer: "", application: "" };

type QA = {
  id: string;
  question: string;
  answer: string;
  status: string;
  source: string;
  bad_answer?: string | null;
  transcript?: string;
  application?: string | null;
};

const SOURCE_STYLE: Record<string, string> = {
  manual: "bg-gray-100 text-gray-600 border-gray-300",
  feedback: "bg-amber-100 text-amber-700 border-amber-300",
  cannot_answer: "bg-rose-100 text-rose-700 border-rose-300",
};

function SourceBadge({ source }: { source: string }) {
  const cls = SOURCE_STYLE[source] ?? "bg-gray-100 text-gray-600 border-gray-300";
  return (
    <span className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide ${cls}`}>
      {source.replace("_", " ")}
    </span>
  );
}

export default function AdminPage() {
  const [token, setToken] = useState("");
  const [items, setItems] = useState<QA[]>([]);
  const [sel, setSel] = useState<QA | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT);

  function startCreate() {
    setSel(null);
    setDraft(EMPTY_DRAFT);
    setCreating(true);
  }

  function selectItem(it: QA) {
    setCreating(false);
    setSel(it);
  }

  async function onCreate(approve: boolean) {
    if (busy || !draft.question.trim()) return;
    if (approve && !draft.answer.trim()) return;
    setBusy(true);
    try {
      const rec = await createQA(token, {
        question: draft.question,
        answer: draft.answer,
        application: draft.application.trim() || null,
      });
      if (approve) await approveQA(token, rec.id);
      setCreating(false);
      setDraft(EMPTY_DRAFT);
      await refresh(token);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    const t = localStorage.getItem("adminToken") || "";
    setToken(t);
    if (t) refresh(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refresh(t: string) {
    setLoading(true);
    try {
      setError("");
      setItems(await listQA(t, "pending"));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function saveToken() {
    localStorage.setItem("adminToken", token);
    refresh(token);
  }

  async function onApprove() {
    if (!sel || busy) return;
    setBusy(true);
    try {
      await editQA(token, sel.id, { answer: sel.answer, application: sel.application ?? null });
      await approveQA(token, sel.id);
      setSel(null);
      await refresh(token);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onReject() {
    if (!sel || busy) return;
    setBusy(true);
    try {
      await rejectQA(token, sel.id);
      setSel(null);
      await refresh(token);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-screen flex-col bg-white text-gray-800">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-gray-200 bg-gray-50 px-4 py-3">
        <h1 className="text-sm font-semibold text-gray-700">Q&amp;A Curation</h1>
        <div className="ml-auto flex items-center gap-2">
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && saveToken()}
            placeholder="Admin token"
            className="w-56 rounded border border-gray-300 px-2 py-1 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
          />
          <button
            onClick={saveToken}
            className="rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-700"
          >
            Load
          </button>
        </div>
      </header>

      {error && (
        <div className="border-b border-rose-200 bg-rose-50 px-4 py-2 text-xs text-rose-700">{error}</div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Pending list */}
        <aside className="flex w-96 flex-col border-r border-gray-200">
          <div className="flex items-center justify-between border-b border-gray-100 px-4 py-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
              Pending ({items.length})
            </span>
            <div className="flex items-center gap-3">
              <button
                onClick={startCreate}
                disabled={!token}
                className="text-xs font-medium text-blue-600 hover:text-blue-700 disabled:opacity-40"
              >
                + New Q&amp;A
              </button>
              <button
                onClick={() => refresh(token)}
                disabled={!token || loading}
                className="text-xs text-gray-400 hover:text-gray-600 disabled:opacity-40"
              >
                {loading ? "…" : "Refresh"}
              </button>
            </div>
          </div>
          <ul className="flex-1 overflow-y-auto p-2">
            {items.length === 0 && !loading && (
              <li className="px-2 py-6 text-center text-xs text-gray-400">
                {token ? "Không có câu hỏi nào đang chờ." : "Nhập admin token để bắt đầu."}
              </li>
            )}
            {items.map((it) => {
              const active = sel?.id === it.id;
              return (
                <li key={it.id}>
                  <button
                    onClick={() => selectItem(it)}
                    className={`mb-1 w-full rounded-md border px-3 py-2 text-left transition-colors ${
                      active
                        ? "border-blue-400 bg-blue-50 ring-1 ring-blue-300"
                        : "border-transparent hover:border-gray-200 hover:bg-gray-50"
                    }`}
                  >
                    <SourceBadge source={it.source} />
                    <p className="mt-1 line-clamp-2 text-sm text-gray-700">{it.question}</p>
                  </button>
                </li>
              );
            })}
          </ul>
        </aside>

        {/* Detail */}
        <section className="flex-1 overflow-y-auto p-6">
          {creating ? (
            <div className="mx-auto max-w-3xl space-y-4">
              <div className="flex items-center gap-2">
                <SourceBadge source="manual" />
              </div>
              <h2 className="text-lg font-semibold text-gray-900">Tạo Q&amp;A mới</h2>

              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Câu hỏi
                </label>
                <textarea
                  value={draft.question}
                  onChange={(e) => setDraft({ ...draft, question: e.target.value })}
                  rows={3}
                  placeholder="Câu hỏi người dùng thường hỏi…"
                  className="w-full rounded-md border border-gray-300 p-3 text-sm leading-relaxed focus:outline-none focus:ring-1 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Câu trả lời đúng
                </label>
                <textarea
                  value={draft.answer}
                  onChange={(e) => setDraft({ ...draft, answer: e.target.value })}
                  rows={8}
                  placeholder="Câu trả lời (bắt buộc nếu muốn duyệt ngay)…"
                  className="w-full rounded-md border border-gray-300 p-3 text-sm leading-relaxed focus:outline-none focus:ring-1 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Application <span className="font-normal normal-case text-gray-400">(tùy chọn)</span>
                </label>
                <input
                  value={draft.application}
                  onChange={(e) => setDraft({ ...draft, application: e.target.value })}
                  placeholder="ví dụ: Phòng thí nghiệm"
                  className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
                />
              </div>

              <div className="flex items-center gap-2 pt-2">
                <button
                  onClick={() => onCreate(true)}
                  disabled={!draft.question.trim() || !draft.answer.trim() || busy}
                  className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {busy ? "Đang xử lý…" : "Tạo & Duyệt"}
                </button>
                <button
                  onClick={() => onCreate(false)}
                  disabled={!draft.question.trim() || busy}
                  className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-40"
                >
                  Lưu nháp
                </button>
                <button
                  onClick={() => setCreating(false)}
                  disabled={busy}
                  className="rounded-md px-4 py-2 text-sm font-medium text-gray-400 hover:text-gray-600 disabled:opacity-40"
                >
                  Hủy
                </button>
              </div>
            </div>
          ) : sel ? (
            <div className="mx-auto max-w-3xl space-y-4">
              <div className="flex items-center gap-2">
                <SourceBadge source={sel.source} />
              </div>
              <h2 className="text-lg font-semibold leading-snug text-gray-900">{sel.question}</h2>

              {sel.bad_answer && (
                <div className="rounded-md border border-rose-200 bg-rose-50 p-3">
                  <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-rose-600">
                    Câu trả lời sai
                  </p>
                  <p className="whitespace-pre-wrap text-sm text-rose-900">{sel.bad_answer}</p>
                </div>
              )}

              {sel.transcript && (
                <details className="rounded-md border border-gray-200">
                  <summary className="cursor-pointer px-3 py-2 text-xs font-medium text-gray-500 hover:text-gray-700">
                    Transcript
                  </summary>
                  <pre className="overflow-x-auto whitespace-pre-wrap border-t border-gray-100 bg-gray-50 p-3 text-xs text-gray-600">
                    {sel.transcript}
                  </pre>
                </details>
              )}

              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Câu trả lời đúng
                </label>
                <textarea
                  value={sel.answer}
                  onChange={(e) => setSel({ ...sel, answer: e.target.value })}
                  rows={10}
                  placeholder="Nhập câu trả lời đúng để CS phê duyệt…"
                  className="w-full rounded-md border border-gray-300 p-3 text-sm leading-relaxed focus:outline-none focus:ring-1 focus:ring-blue-400"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Application <span className="font-normal normal-case text-gray-400">(tùy chọn)</span>
                </label>
                <input
                  value={sel.application ?? ""}
                  onChange={(e) => setSel({ ...sel, application: e.target.value })}
                  placeholder="ví dụ: Phòng thí nghiệm"
                  className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
                />
              </div>

              <div className="flex items-center gap-2 pt-2">
                <button
                  onClick={onApprove}
                  disabled={!sel.answer.trim() || busy}
                  className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {busy ? "Đang xử lý…" : "Approve"}
                </button>
                <button
                  onClick={onReject}
                  disabled={busy}
                  className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-rose-600 hover:bg-rose-50 disabled:opacity-40"
                >
                  Reject
                </button>
              </div>
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-gray-400">
              Chọn một câu hỏi bên trái để xử lý.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
