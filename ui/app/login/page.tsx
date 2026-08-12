"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { login } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [userName, setUserName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (busy || !userName.trim() || !password) return;
    setBusy(true);
    setError("");
    try {
      const res = await login(userName.trim(), password);
      router.replace(res.role === "admin" ? "/admin" : "/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Đăng nhập thất bại");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-gray-50">
      <form
        onSubmit={onSubmit}
        className="w-80 rounded-lg border border-gray-200 bg-white p-6 shadow-sm"
      >
        <h1 className="mb-1 text-lg font-semibold text-gray-800">Đăng nhập</h1>
        <p className="mb-5 text-xs text-gray-500">Hỗ trợ phần mềm CenLab</p>

        <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-gray-500">
          Tên đăng nhập
        </label>
        <input
          value={userName}
          onChange={(e) => setUserName(e.target.value)}
          autoFocus
          autoComplete="username"
          className="mb-4 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
        />

        <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-gray-500">
          Mật khẩu
        </label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
          className="mb-4 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
        />

        {error && (
          <p className="mb-3 rounded border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy || !userName.trim() || !password}
          className="w-full rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "Đang đăng nhập…" : "Đăng nhập"}
        </button>
      </form>
    </div>
  );
}
