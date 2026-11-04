"use client";

import { useState } from "react";
import { ExportKind, downloadExport } from "@/lib/api";

/** A local date as `YYYY-MM-DD`. Not `toISOString()`, which is UTC and can be a day off. */
function ymd(d: Date): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

const inputCls = "rounded border border-gray-300 bg-white px-2 py-0.5 text-xs";

/**
 * Admin only: export one customer's bugs or feature requests, created in a date range,
 * to Excel. The default range is this month so far; clearing a date leaves that end open.
 */
export default function ExportBar({ customerId, kind }: { customerId: string; kind: ExportKind }) {
  const today = new Date();
  const [from, setFrom] = useState(ymd(new Date(today.getFullYear(), today.getMonth(), 1)));
  const [to, setTo] = useState(ymd(today));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const badRange = from !== "" && to !== "" && from > to;

  async function run() {
    setBusy(true);
    try {
      setError("");
      await downloadExport(customerId, kind, from, to);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-gray-200 bg-gray-50 px-4 py-2 text-xs text-gray-600">
      <span>Ngày tạo từ</span>
      <input
        type="date"
        value={from}
        onChange={(e) => setFrom(e.target.value)}
        className={inputCls}
      />
      <span>đến</span>
      <input type="date" value={to} onChange={(e) => setTo(e.target.value)} className={inputCls} />
      <button
        onClick={run}
        disabled={busy || badRange}
        className="rounded bg-emerald-600 px-3 py-1 font-medium text-white hover:bg-emerald-700 disabled:opacity-40"
      >
        {busy ? "Đang xuất…" : "Xuất Excel"}
      </button>
      {badRange && <span className="text-rose-700">Từ ngày phải trước đến ngày.</span>}
      {error && <span className="text-rose-700">{error}</span>}
    </div>
  );
}
