"use client";

import { useEffect, useState } from "react";
import {
  Customer,
  Role,
  createCustomer,
  listApplications,
  listCustomers,
  updateCustomer,
} from "@/lib/api";
import ApplicationsSelect from "./ApplicationsSelect";

type Draft = {
  customer_id: string;
  name: string;
  password: string;
  role: Role;
  enabled_applications: string[];
};

const EMPTY_DRAFT: Draft = {
  customer_id: "",
  name: "",
  password: "",
  role: "user",
  enabled_applications: [],
};

function RoleBadge({ role }: { role: Role }) {
  const cls =
    role === "admin"
      ? "bg-violet-100 text-violet-700 border-violet-300"
      : "bg-gray-100 text-gray-600 border-gray-300";
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide ${cls}`}
    >
      {role}
    </span>
  );
}

export default function CustomerAdmin() {
  const [items, setItems] = useState<Customer[]>([]);
  const [appOptions, setAppOptions] = useState<string[]>([]);
  const [sel, setSel] = useState<Customer | null>(null);
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT);
  const [newPassword, setNewPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    refresh();
    // The catalogue is static per deployment, so it is fetched once and never with
    // the customer list. A failure here only costs the dropdown its options — the
    // form still works on whatever a customer already has, so it isn't surfaced as
    // a form error.
    listApplications()
      .then((apps) => setAppOptions(apps.map((a) => a.name)))
      .catch(() => setAppOptions([]));
  }, []);

  async function refresh() {
    setLoading(true);
    try {
      setError("");
      setItems(await listCustomers());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function startCreate() {
    setSel(null);
    setDraft(EMPTY_DRAFT);
    setCreating(true);
  }

  function selectItem(c: Customer) {
    setCreating(false);
    setNewPassword("");
    setSel(c);
  }

  async function onCreate() {
    if (busy || !draft.customer_id.trim() || !draft.name.trim() || draft.password.length < 8) return;
    setBusy(true);
    try {
      setError("");
      await createCustomer({
        customer_id: draft.customer_id.trim(),
        name: draft.name.trim(),
        password: draft.password,
        role: draft.role,
        enabled_applications: draft.enabled_applications,
      });
      setCreating(false);
      setDraft(EMPTY_DRAFT);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onSave() {
    if (!sel || busy) return;
    if (newPassword && newPassword.length < 8) {
      setError("Mật khẩu phải có ít nhất 8 ký tự");
      return;
    }
    setBusy(true);
    try {
      setError("");
      await updateCustomer(sel.customer_id, {
        name: sel.name,
        role: sel.role,
        enabled_applications: sel.enabled_applications,
        // Only send a password when one was typed — an empty field must leave the
        // existing credentials alone, not wipe them.
        ...(newPassword ? { password: newPassword } : {}),
      });
      setNewPassword("");
      setSel(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const labelCls =
    "mb-1 block text-xs font-semibold uppercase tracking-wide text-gray-500";
  const inputCls =
    "w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400";

  return (
    <>
      {error && (
        <div className="border-b border-rose-200 bg-rose-50 px-4 py-2 text-xs text-rose-700">
          {error}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        <aside className="flex w-96 flex-col border-r border-gray-200">
          <div className="flex items-center justify-between border-b border-gray-100 px-4 py-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
              Khách hàng ({items.length})
            </span>
            <div className="flex items-center gap-3">
              <button
                onClick={startCreate}
                className="text-xs font-medium text-blue-600 hover:text-blue-700"
              >
                + Tạo mới
              </button>
              <button
                onClick={refresh}
                disabled={loading}
                className="text-xs text-gray-400 hover:text-gray-600 disabled:opacity-40"
              >
                {loading ? "…" : "Refresh"}
              </button>
            </div>
          </div>
          <ul className="flex-1 overflow-y-auto p-2">
            {items.length === 0 && !loading && (
              <li className="px-2 py-6 text-center text-xs text-gray-400">
                Chưa có khách hàng nào.
              </li>
            )}
            {items.map((c) => (
              <li key={c.customer_id}>
                <button
                  onClick={() => selectItem(c)}
                  className={`mb-1 w-full rounded-md border px-3 py-2 text-left transition-colors ${
                    sel?.customer_id === c.customer_id
                      ? "border-blue-400 bg-blue-50 ring-1 ring-blue-300"
                      : "border-transparent hover:border-gray-200 hover:bg-gray-50"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <RoleBadge role={c.role} />
                    {!c.has_password && (
                      <span className="rounded-full border border-amber-300 bg-amber-100 px-2 py-0.5 text-[10px] font-medium uppercase text-amber-700">
                        chưa có mật khẩu
                      </span>
                    )}
                  </div>
                  <p className="mt-1 text-sm text-gray-700">{c.name}</p>
                  <p className="font-mono text-xs text-gray-400">{c.customer_id}</p>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <section className="flex-1 overflow-y-auto p-6">
          {creating ? (
            <div className="mx-auto max-w-3xl space-y-4">
              <h2 className="text-lg font-semibold text-gray-900">Tạo khách hàng mới</h2>

              <div>
                <label className={labelCls}>Tên đăng nhập (customer_id)</label>
                <input
                  value={draft.customer_id}
                  onChange={(e) => setDraft({ ...draft, customer_id: e.target.value })}
                  placeholder="ví dụ: ttp"
                  className={`${inputCls} font-mono`}
                />
                {/* This is the one irreversible field on the form; the constraint is a
                    consequence of the login name and the tenant id being the same value. */}
                <p className="mt-1 text-xs text-amber-700">
                  Đây cũng là <strong>tên đăng nhập</strong> và <strong>không thể đổi</strong> sau
                  khi tạo. Chỉ dùng chữ, số, dấu chấm, gạch dưới hoặc gạch ngang.
                </p>
              </div>

              <div>
                <label className={labelCls}>Tên hiển thị</label>
                <input
                  value={draft.name}
                  onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                  placeholder="Công ty TNHH…"
                  className={inputCls}
                />
              </div>

              <div>
                <label className={labelCls}>Mật khẩu</label>
                <input
                  type="password"
                  value={draft.password}
                  onChange={(e) => setDraft({ ...draft, password: e.target.value })}
                  className={inputCls}
                />
                <p className="mt-1 text-xs text-gray-400">Ít nhất 8 ký tự, tối đa 72 byte.</p>
              </div>

              <div>
                <label className={labelCls}>Vai trò</label>
                <select
                  value={draft.role}
                  onChange={(e) => setDraft({ ...draft, role: e.target.value as Role })}
                  className={inputCls}
                >
                  <option value="user">user</option>
                  <option value="admin">admin</option>
                </select>
              </div>

              <div>
                <label className={labelCls}>Applications</label>
                <ApplicationsSelect
                  value={draft.enabled_applications}
                  options={appOptions}
                  onChange={(apps) => setDraft({ ...draft, enabled_applications: apps })}
                  disabled={busy}
                />
              </div>

              <div className="flex items-center gap-2 pt-2">
                <button
                  onClick={onCreate}
                  disabled={
                    busy ||
                    !draft.customer_id.trim() ||
                    !draft.name.trim() ||
                    draft.password.length < 8
                  }
                  className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {busy ? "Đang tạo…" : "Tạo"}
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
                <RoleBadge role={sel.role} />
              </div>
              <h2 className="text-lg font-semibold text-gray-900">
                {sel.name}
                <span className="ml-2 font-mono text-sm font-normal text-gray-400">
                  {sel.customer_id}
                </span>
              </h2>

              <div>
                <label className={labelCls}>Tên hiển thị</label>
                <input
                  value={sel.name}
                  onChange={(e) => setSel({ ...sel, name: e.target.value })}
                  className={inputCls}
                />
              </div>

              <div>
                <label className={labelCls}>Vai trò</label>
                <select
                  value={sel.role}
                  onChange={(e) => setSel({ ...sel, role: e.target.value as Role })}
                  className={inputCls}
                >
                  <option value="user">user</option>
                  <option value="admin">admin</option>
                </select>
              </div>

              <div>
                <label className={labelCls}>Applications</label>
                <ApplicationsSelect
                  value={sel.enabled_applications}
                  options={appOptions}
                  onChange={(apps) => setSel({ ...sel, enabled_applications: apps })}
                  disabled={busy}
                />
              </div>

              <div>
                <label className={labelCls}>Đặt lại mật khẩu</label>
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="Để trống nếu không đổi"
                  className={inputCls}
                />
              </div>

              <div className="flex items-center gap-2 pt-2">
                <button
                  onClick={onSave}
                  disabled={busy || !sel.name.trim()}
                  className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {busy ? "Đang lưu…" : "Lưu"}
                </button>
                <button
                  onClick={() => setSel(null)}
                  disabled={busy}
                  className="rounded-md px-4 py-2 text-sm font-medium text-gray-400 hover:text-gray-600 disabled:opacity-40"
                >
                  Hủy
                </button>
              </div>
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-gray-400">
              Chọn một khách hàng bên trái, hoặc tạo mới.
            </div>
          )}
        </section>
      </div>
    </>
  );
}
