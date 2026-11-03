"use client";

import { useEffect, useRef, useState } from "react";
import {
  FEATURE_CATEGORIES,
  FeatureCategory,
  RequestDetail,
  RequestPatch,
  RequestStatus,
  RequestSummary,
} from "@/lib/api";
import { parseTranscript } from "@/lib/transcript";
import MessageList from "@/components/MessageList";
import IssueStatusBadge, {
  FEATURE_CATEGORY_LABEL,
  RequestKind,
  STATUS_OPTIONS,
} from "@/components/IssueStatusBadge";

/** The words that differ between the bug list and the feature-request list. */
const NOUN: Record<
  RequestKind,
  { list: string; one: string; createdAt: string; note: string; notePlaceholder: string }
> = {
  bug: {
    list: "Sự cố",
    one: "sự cố",
    createdAt: "Báo lúc",
    note: "Ghi chú từ CS",
    notePlaceholder: "Ghi chú cho khách hàng (không bắt buộc)…",
  },
  feature: {
    list: "Yêu cầu",
    one: "yêu cầu",
    createdAt: "Gửi lúc",
    // A feature request's `note` is CS's answer to it.
    note: "Giải pháp",
    notePlaceholder: "Giải pháp đề xuất cho khách hàng…",
  },
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

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
  kind,
  onSave,
}: {
  detail: RequestDetail;
  kind: RequestKind;
  onSave: (patch: RequestPatch) => Promise<void>;
}) {
  const [status, setStatus] = useState<RequestStatus>(detail.status);
  const [dueDate, setDueDate] = useState(detail.due_date ?? "");
  const [note, setNote] = useState(detail.note ?? "");
  const [category, setCategory] = useState<FeatureCategory | "">(detail.category ?? "");
  const [saving, setSaving] = useState(false);

  const patch: RequestPatch = {};
  if (kind === "feature" && category !== (detail.category ?? "")) {
    patch.category = category || null;
  }
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
            onChange={(e) => setStatus(e.target.value as RequestStatus)}
            className="rounded border border-gray-300 bg-white px-2 py-1"
          >
            {STATUS_OPTIONS[kind].statuses.map((s) => (
              <option key={s} value={s}>
                {STATUS_OPTIONS[kind].label[s]}
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
        {kind === "feature" && (
          <label className="flex items-center gap-1">
            Thuộc hạng mục
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value as FeatureCategory | "")}
              className="rounded border border-gray-300 bg-white px-2 py-1"
            >
              <option value="">— Chưa chọn —</option>
              {FEATURE_CATEGORIES.map((c) => (
                <option key={c} value={c}>
                  {FEATURE_CATEGORY_LABEL[c]}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {kind === "feature" && <p className="text-xs text-gray-600">{NOUN.feature.note}</p>}
      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        maxLength={2000}
        rows={3}
        placeholder={NOUN[kind].notePlaceholder}
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

// Method syntax on purpose: it lets a caller pass functions typed with the narrower
// bug or feature types (e.g. `(patch: IssuePatch) => …`).
interface Props {
  /** Bugs ("Sự cố") or feature requests ("Yêu cầu"): picks the statuses and wording. */
  kind?: RequestKind;
  /** Loads the whole list, newest first. Called on mount, on Refresh and when it changes identity. */
  loadList(): Promise<RequestSummary[]>;
  loadDetail(id: string): Promise<RequestDetail>;
  /** Given only to admins: shows the edit form (status, due date, note) on the open item. */
  onUpdate?(id: string, patch: RequestPatch): Promise<RequestDetail>;
  /** Reports the list after every load or status change, e.g. for a tab counter. */
  onItemsChange?(items: RequestSummary[]): void;
  emptyText: string;
  /** Opens this item once the list has loaded, e.g. one the user just created. */
  openId?: string | null;
}

/**
 * List on the left, the open item on the right. Shared by bugs and feature requests,
 * and by the customer's own pages and the admin's per-customer tabs; only `onUpdate`
 * differs between customer and admin.
 */
export default function IssueView({
  kind = "bug",
  loadList,
  loadDetail,
  onUpdate,
  onItemsChange,
  emptyText,
  openId,
}: Props) {
  const noun = NOUN[kind];
  const options = STATUS_OPTIONS[kind];
  const [items, setItems] = useState<RequestSummary[]>([]);
  const [filter, setFilter] = useState<RequestStatus | "all">("all");
  const [listLoading, setListLoading] = useState(false);
  const [selId, setSelId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RequestDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  // The issue most recently clicked. A response for any other id is stale — the user
  // clicked away before it arrived — and is dropped.
  const latest = useRef<string | null>(null);

  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadList]);

  useEffect(() => {
    if (openId) open(openId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openId]);

  function publish(next: RequestSummary[]) {
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

  async function save(patch: RequestPatch) {
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
              {noun.list} ({shown.length})
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
            {options.statuses.map((s) => (
              <button key={s} onClick={() => setFilter(s)} className={chip(filter === s)}>
                {options.label[s]}
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
                      <IssueStatusBadge status={i.status} kind={kind} />
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
                    {detail.title || noun.list}
                  </h2>
                  <IssueStatusBadge status={detail.status} kind={kind} />
                </div>
                {detail.requester && (
                  <p className="text-xs text-gray-600">
                    Người yêu cầu: <b>{detail.requester.name}</b> · {detail.requester.position} ·{" "}
                    {detail.requester.department}
                  </p>
                )}
                <p className="whitespace-pre-wrap text-sm text-gray-700">{detail.summary}</p>
                {detail.reason && (
                  <p className="whitespace-pre-wrap text-sm text-gray-600">
                    <span className="font-medium">Lý do:</span> {detail.reason}
                  </p>
                )}
                {detail.files && detail.files.length > 0 && (
                  <ul className="flex flex-wrap gap-2">
                    {detail.files.map((f) => (
                      <li key={f.url}>
                        <a
                          href={f.url}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-block rounded border border-gray-200 bg-gray-50 px-2 py-1 text-xs text-blue-600 hover:bg-gray-100"
                        >
                          📎 {f.name} ({formatSize(f.size_bytes)})
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
                {!onUpdate && detail.category && (
                  <p className="text-xs text-gray-600">
                    Thuộc hạng mục: <b>{FEATURE_CATEGORY_LABEL[detail.category]}</b>
                  </p>
                )}
                {onUpdate ? (
                  <IssueEditor
                    // Remount after a save so the drafts restart from the saved values.
                    key={`${detail.id}|${detail.status}|${detail.due_date}|${detail.note}|${detail.category}`}
                    detail={detail}
                    kind={kind}
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
                          {noun.note}
                        </p>
                        <p className="mt-1 whitespace-pre-wrap text-sm text-gray-800">{detail.note}</p>
                      </div>
                    )}
                  </>
                )}
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-gray-400">
                  <span>
                    {noun.createdAt} {formatTime(detail.created_at)}
                  </span>
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
              {/* A feature request was written in a form: it has no conversation at all. */}
              {kind === "bug" &&
                (detail.transcript?.trim() ? (
                  <MessageList messages={parseTranscript(detail.transcript)} loading={false} />
                ) : (
                  <div className="flex flex-1 items-center justify-center text-sm text-gray-400">
                    Không có nội dung hội thoại.
                  </div>
                ))}
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center text-sm text-gray-400">
              {detailLoading ? "Đang tải…" : `Chọn một ${noun.one} để xem chi tiết.`}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
