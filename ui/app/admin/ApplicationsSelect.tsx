"use client";

import { useEffect, useMemo, useRef, useState } from "react";

interface Props {
  /** Currently enabled applications, as display names. */
  value: string[];
  /** The catalogue from GET /admin/applications, as display names. */
  options: string[];
  onChange: (next: string[]) => void;
  disabled?: boolean;
}

/**
 * Checkbox dropdown for `enabled_applications`.
 *
 * The stored value is a list of display names ("Lấy mẫu - Quan trắc") — the same
 * strings the catalogue serves — so selection is plain string equality and no
 * slug translation happens here; the server does that when it filters Qdrant.
 *
 * A value that isn't in the catalogue is still rendered (flagged, at the end of
 * the list) rather than dropped. Otherwise editing an unrelated field on a
 * customer holding a retired or misspelled application would silently delete it
 * on save, since the form submits the whole list.
 */
export default function ApplicationsSelect({ value, options, onChange, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // Unknown values keep their place in the list so they can be unchecked.
  const unknown = useMemo(() => value.filter((v) => !options.includes(v)), [value, options]);
  const all = useMemo(() => [...options, ...unknown], [options, unknown]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return all;
    return all.filter((a) => a.toLowerCase().includes(q));
  }, [all, query]);

  const toggle = (app: string) => {
    onChange(value.includes(app) ? value.filter((a) => a !== app) : [...value, app]);
  };

  return (
    <div ref={boxRef} className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 rounded-md border border-gray-300 px-3 py-2 text-left text-sm focus:outline-none focus:ring-1 focus:ring-blue-400 disabled:opacity-50"
      >
        <span className="flex flex-1 flex-wrap gap-1">
          {value.length === 0 ? (
            <span className="text-gray-400">Chọn applications…</span>
          ) : (
            value.map((app) => (
              <span
                key={app}
                className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs ${
                  options.includes(app)
                    ? "border-blue-300 bg-blue-50 text-blue-700"
                    : "border-amber-300 bg-amber-50 text-amber-700"
                }`}
              >
                {app}
                <span
                  role="button"
                  tabIndex={-1}
                  aria-label={`Bỏ ${app}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggle(app);
                  }}
                  className="cursor-pointer text-current opacity-50 hover:opacity-100"
                >
                  ×
                </span>
              </span>
            ))
          )}
        </span>
        <span className="shrink-0 text-xs text-gray-400">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="absolute z-20 mt-1 w-full rounded-md border border-gray-200 bg-white shadow-lg">
          <div className="border-b border-gray-100 p-2">
            <input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Tìm application…"
              className="w-full rounded border border-gray-200 px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
            />
          </div>

          <ul className="max-h-64 overflow-y-auto py-1">
            {visible.length === 0 && (
              <li className="px-3 py-3 text-center text-xs text-gray-400">Không có kết quả.</li>
            )}
            {visible.map((app) => {
              const checked = value.includes(app);
              return (
                <li key={app}>
                  <label className="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm hover:bg-gray-50">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggle(app)}
                      className="h-4 w-4 accent-blue-600"
                    />
                    <span className={checked ? "text-gray-900" : "text-gray-600"}>{app}</span>
                    {!options.includes(app) && (
                      <span className="ml-auto text-[10px] uppercase text-amber-600">
                        không xác định
                      </span>
                    )}
                  </label>
                </li>
              );
            })}
          </ul>

          <div className="flex items-center justify-between border-t border-gray-100 px-3 py-1.5 text-xs">
            <button
              type="button"
              onClick={() => onChange(options)}
              className="text-blue-600 hover:text-blue-700"
            >
              Chọn tất cả
            </button>
            <span className="text-gray-400">{value.length} đã chọn</span>
            <button
              type="button"
              onClick={() => onChange([])}
              className="text-gray-400 hover:text-gray-600"
            >
              Bỏ chọn
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
