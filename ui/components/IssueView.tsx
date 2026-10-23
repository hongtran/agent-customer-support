"use client";

import { useEffect, useRef, useState } from "react";
import { ISSUE_STATUSES, IssueDetail, IssuePatch, IssueStatus, IssueSummary } from "@/lib/api";
import { parseTranscript } from "@/lib/transcript";
import MessageList from "@/components/MessageList";
import IssueStatusBadge, { ISSUE_STATUS_LABEL } from "@/components/IssueStatusBadge";

function formatTime(iso: string | null): string {
  if (!iso) return "";
  return new Date(iso).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
}

/**
 * "YYYY-MM-DD" → "dd/mm/yyyy", from the string itself. `new Date("2026-11-05")` parses as
 * UTC midnight, which shows the day before in any time zone west of UTC.
 */
function formatDay(day: string): string {
  const [y, m, d] = day.split("-");
  return `${d}/${m}/${y}`;
}

/**
 * The admin's edit form for the open issue. Nothing is saved until "Lưu", and only
 * changed fields are sent — so saving a note alone adds no status-history entry.
 * The parent remounts it (via `key`) after each save, which resets the drafts to the
 * saved values.
 */
function IssueEditor({
  detail,
  onSave,
}: {
  detail: IssueDetail;
  onSave: (patch: IssuePatch) => Promise<void>;
}) {
  const [status, setStatus] = useState<IssueStatus>(detail.status);
  const [dueDate, setDueDate] = useState(detail.due_date ?? "");
  const [note, setNote] = useState(detail.note ?? "");
  const [saving, setSaving] = useState(false);

  const patch: IssuePatch = {};
  if (status !== detail.status) patch.status = status;
  if (dueDate !== (detail.due_date ?? "")) patch.due_date = dueDate || null;
  if (note.trim() !== (detail.note ?? "")) patch.note = note.trim() || null;
  const dirty = Object.keys(patch).length > 0;

  async function save() {
    setSaving(true);
    try {
      await onSave(patch);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-2 rounded-md border border-gray-200 bg-gray-50 p-3">
      <div className="flex flex-wrap items-center gap-3 text-xs text-gray-600">
        <label className="flex items-center gap-1">
          Trạng thái
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as IssueStatus)}
            className="rounded border border-gray-300 bg-white px-2 py-1"
          >
            {ISSUE_STATUSES.map((s) => (
              <option key={s} value={s}>
                {ISSUE_STATUS_LABEL[s]}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1">
          Dự kiến hoàn thành
          <input
            type="date"
            value={dueDate}
            onChange={(e) => setDueDate(e.target.value)}
            className="rounded border border-gray-300 bg-white px-2 py-1"
          />
        </label>
      </div>
      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        maxLength={2000}
        rows={3}
        placeholder="Ghi chú cho khách hàng (không bắt buộc)…"
        className="w-full rounded border border-gray-300 bg-white px-2 py-1 text-sm"
      />
      <div className="flex justify-end">
        <button
          onClick={save}
          disabled={!dirty || saving}
          className="rounded bg-blue-600 px-4 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-40"
        >
          {saving ? "Đang lưu…" : "Lưu"}
        </button>
      </div>
    </div>
  );
}

interface Props {
  /** Loads the whole list, newest first. Called on mount and on Refresh. */
  loadList: () => Promise<IssueSummary[]>;
  loadDetail: (id: string) => Promise<IssueDetail>;
  /** Given only to admins: shows the edit form (status, due date, note) on the open issue. */
  onUpdate?: (id: string, patch: IssuePatch) => Promise<IssueDetail>;
  /** Reports the list after every load or status change, e.g. for a tab counter. */
  onItemsChange?: (items: IssueSummary[]) => void;
  emptyText: string;
}

/**
 * Issue list on the left, the open issue on the right. Shared by the customer's own
 * page and the admin's per-customer tab; only `onUpdate` differs between them.
 */
export default function IssueView({
  loadList,
  loadDetail,
  onUpdate,
  onItemsChange,
  emptyText,
}: Props) {
  const [items, setItems] = useState<IssueSummary[]>([]);
  const [filter, setFilter] = useState<IssueStatus | "all">("all");
  const [listLoading, setListLoading] = useState(false);
  const [selId, setSelId] = useState<string | null>(null);
  const [detail, setDetail] = useState<IssueDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  // The issue most recently clicked. A response for any other id is stale — the user
  // clicked away before it arrived — and is dropped.
  const latest = useRef<string | null>(null);

  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadList]);

  function publish(next: IssueSummary[]) {
    setItems(next);
    onItemsChange?.(next);
  }

  async function reload() {
    setListLoading(true);
    try {
      setError("");
      publish(await loadList());
    } catch (e) {
      setError(String(e));
    } finally {
      setListLoading(false);
    }
  }

  async function open(id: string) {
    latest.current = id;
    setSelId(id);
    setDetail(null);
    setDetailLoading(true);
    try {
      setError("");
      const d = await loadDetail(id);
      if (latest.current === id) setDetail(d);
    } catch (e) {
      if (latest.current === id) setError(String(e));
    } finally {
      if (latest.current === id) setDetailLoading(false);
    }
  }

  async function save(patch: IssuePatch) {
    if (!onUpdate || !detail) return;
    try {
      setError("");
      const updated = await onUpdate(detail.id, patch);
      // Patch both views in place rather than reloading: the list keeps its scroll
      // position and the transcript is not re-fetched.
      publish(
        items.map((i) =>
          i.id === updated.id
            ? { ...i, status: updated.status, status_updated_at: updated.status_updated_at }
            : i,
        ),
      );
      setDetail((d) => (d && d.id === updated.id ? updated : d));
    } catch (e) {
      setError(String(e));
    }
  }

  const shown = filter === "all" ? items : items.filter((i) => i.status === filter);
  const chip = (active: boolean) =>
    `rounded-full px-2.5 py-0.5 text-[11px] ${
      active ? "bg-blue-100 text-blue-700" : "text-gray-500 hover:bg-gray-100"
    }`;

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {error && (
        <div className="border-b border-rose-200 bg-rose-50 px-4 py-2 text-xs text-rose-700">{error}</div>
      )}
      <div className="flex flex-1 overflow-hidden">
        <aside className="flex w-96 flex-col border-r border-gray-200">
          <div className="flex items-center justify-between border-b border-gray-100 px-4 py-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
              Sự cố ({shown.length})
            </span>
            <button
              onClick={reload}
              disabled={listLoading}
              className="text-xs text-gray-400 hover:text-gray-600 disabled:opacity-40"
            >
              {listLoading ? "…" : "Refresh"}
            </button>
          </div>
          <div className="flex flex-wrap gap-1 border-b border-gray-100 px-3 py-2">
            <button onClick={() => setFilter("all")} className={chip(filter === "all")}>
              Tất cả
            </button>
            {ISSUE_STATUSES.map((s) => (
              <button key={s} onClick={() => setFilter(s)} className={chip(filter === s)}>
                {ISSUE_STATUS_LABEL[s]}
              </button>
            ))}
          </div>
          <ul className="flex-1 overflow-y-auto p-2">
            {shown.length === 0 && !listLoading && (
              <li className="px-2 py-6 text-center text-xs text-gray-400">{emptyText}</li>
            )}
            {shown.map((i) => {
              const active = selId === i.id;
              return (
                <li key={i.id}>
                  <button
                    onClick={() => open(i.id)}
                    className={`mb-1 w-full rounded-md border px-3 py-2 text-left transition-colors ${
                      active
                        ? "border-blue-400 bg-blue-50 ring-1 ring-blue-300"
                        : "border-transparent hover:border-gray-200 hover:bg-gray-50"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <p className="line-clamp-2 text-sm text-gray-700">{i.title || i.summary}</p>
                      <IssueStatusBadge status={i.status} />
                    </div>
                    <p className="mt-1 text-[11px] text-gray-400">
                      {formatTime(i.created_at)}
                      {i.mantis_issue_id != null && ` · #${i.mantis_issue_id}`}
                    </p>
                  </button>
                </li>
              );
            })}
          </ul>
        </aside>

        <section className="flex flex-1 flex-col overflow-hidden">
          {detail ? (
            <>
              <div className="space-y-2 border-b border-gray-100 px-4 py-3">
                <div className="flex flex-wrap items-center gap-3">
                  <h2 className="text-base font-semibold text-gray-900">
                    {detail.title || "Sự cố"}
                  </h2>
                  <IssueStatusBadge status={detail.status} />
                </div>
                <p className="whitespace-pre-wrap text-sm text-gray-700">{detail.summary}</p>
                {onUpdate ? (
                  <IssueEditor
                    // Remount after a save so the drafts restart from the saved values.
                    key={`${detail.id}|${detail.status}|${detail.due_date}|${detail.note}`}
                    detail={detail}
                    onSave={save}
                  />
                ) : (
                  <>
                    {detail.due_date && (
                      <p className="text-xs text-gray-600">
                        Dự kiến hoàn thành: <b>{formatDay(detail.due_date)}</b>
                      </p>
                    )}
                    {detail.note && (
                      <div className="rounded-md border border-blue-200 bg-blue-50 px-3 py-2">
                        <p className="text-[11px] font-semibold uppercase tracking-wide text-blue-700">
                          Ghi chú từ CS
                        </p>
                        <p className="mt-1 whitespace-pre-wrap text-sm text-gray-800">{detail.note}</p>
                      </div>
                    )}
                  </>
                )}
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-gray-400">
                  <span>Báo lúc {formatTime(detail.created_at)}</span>
                  {detail.status_updated_at && (
                    <span>Cập nhật {formatTime(detail.status_updated_at)}</span>
                  )}
                  {detail.application && <span>Ứng dụng: {detail.application}</span>}
                  {detail.mantis_issue_id != null &&
                    // The tracker is internal: only admins get a link into it.
                    (onUpdate && detail.mantis_issue_url ? (
                      <a
                        href={detail.mantis_issue_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-blue-600 hover:text-blue-700"
                      >
                        MantisBT #{detail.mantis_issue_id}
                      </a>
                    ) : (
                      <span>MantisBT #{detail.mantis_issue_id}</span>
                    ))}
                </div>
              </div>
              {detail.transcript.trim() ? (
                <MessageList messages={parseTranscript(detail.transcript)} loading={false} />
              ) : (
                <div className="flex flex-1 items-center justify-center text-sm text-gray-400">
                  Không có nội dung hội thoại.
                </div>
              )}
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center text-sm text-gray-400">
              {detailLoading ? "Đang tải…" : "Chọn một sự cố để xem chi tiết."}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
