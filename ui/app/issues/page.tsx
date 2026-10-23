"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { getMyIssue, listMyIssues } from "@/lib/api";
import { logout, useSession } from "@/lib/useSession";
import IssueView from "@/components/IssueView";

/** The bugs this customer reported through the agent, and where CS has them. */
export default function MyIssuesPage() {
  const router = useRouter();
  const session = useSession();

  if (session.status === "loading") {
    return <div className="flex h-screen items-center justify-center text-sm text-gray-400">…</div>;
  }

  return (
    <div className="flex h-screen flex-col bg-white text-gray-800">
      <header className="flex items-center gap-3 border-b border-gray-200 bg-gray-50 px-4 py-3">
        <Link href="/" className="text-xs text-blue-600 hover:text-blue-700">
          ← Trang chat
        </Link>
        <h1 className="text-sm font-semibold text-gray-700">Sự cố đã báo</h1>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs text-gray-500">{session.me.name}</span>
          <button
            onClick={() => logout(router)}
            className="rounded px-3 py-1 text-xs text-gray-500 hover:text-gray-700"
          >
            Đăng xuất
          </button>
        </div>
      </header>
      <IssueView
        loadList={listMyIssues}
        loadDetail={getMyIssue}
        emptyText="Bạn chưa báo sự cố nào."
      />
    </div>
  );
}
