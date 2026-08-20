"use client";

import { useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { login } from "@/lib/api";
import loginBg from "@/public/login-bg.jpg";

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
    // The navy matches the artwork's darkest tone, so the page paints the right
    // colour immediately instead of flashing the white body underneath.
    <div className="relative flex h-screen items-center justify-center overflow-hidden bg-[#04122b] px-4">
      {/* Decorative, so alt="" keeps it out of the accessibility tree. The static
          import hands next/image the real dimensions and a blur placeholder, and
          `priority` marks it as this page's LCP element. next/image also resizes
          and re-encodes the 3344px source per viewport, which is what keeps a
          706 KB JPEG from being downloaded in full on a phone.

          object-left below `sm`: the art is 16:9 with its subject bottom-left, so a
          phone crops it to a tall slice. Centring that slice lands on the empty middle
          and the background reads as flat navy; anchoring left keeps the glassware in
          frame. From `sm` up the viewport is wide enough to centre the whole scene. */}
      <Image
        src={loginBg}
        alt=""
        fill
        priority
        placeholder="blur"
        sizes="100vw"
        className="object-cover object-left sm:object-center"
      />

      <form
        onSubmit={onSubmit}
        className="relative w-80 rounded-lg border border-gray-200 bg-white p-6 shadow-2xl"
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
