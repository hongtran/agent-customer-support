"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ConversationDetail,
  ConversationSummary,
  getCustomerConversation,
  listCustomerConversations,
} from "@/lib/api";
import { logout, useSession } from "@/lib/useSession";
import MessageList, { type Message } from "@/components/MessageList";

function formatTime(iso: string | null): string {
  if (!iso) return "";
  return new Date(iso).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
}

/**
 * Stored turns → the chat view's message shape. No `messageId`, so MessageList shows
 * no vote/copy buttons: this is a read-only transcript, and a vote from CS would be
 * stored as if the customer had cast it.
 */
function toMessages(detail: ConversationDetail): Message[] {
  return detail.turns.map((t) => ({
    role: t.role === "user" ? "user" : "agent",
    content: t.content,
    attachments: t.attachments.map((a) => ({ media_type: a.media_type, url: a.url })),
  }));
}

export default function CustomerConversationsPage() {
  const router = useRouter();
  const session = useSession();
  const { customerId: rawId } = useParams<{ customerId: string }>();
  const customerId = decodeURIComponent(rawId);

  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [listLoading, setListLoading] = useState(false);
  const [selId, setSelId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  // The conversation most recently clicked. A response for any other id is stale — the
  // user clicked away before it arrived — and is dropped.
  const latest = useRef<string | null>(null);

  const isAdmin = session.status === "ready" && session.me.role === "admin";

  useEffect(() => {
    if (isAdmin) loadPage(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAdmin, customerId]);

  /** `from` = null reloads from the top; a cursor appends the next page. */
  async function loadPage(from: string | null) {
    setListLoading(true);
    try {
      setError("");
      const page = await listCustomerConversations(customerId, from);
      setItems((prev) => (from ? [...prev, ...page.items] : page.items));
      setCursor(page.next_cursor);
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
      const d = await getCustomerConversation(customerId, id);
      if (latest.current === id) setDetail(d);
    } catch (e) {
      if (latest.current === id) setError(String(e));
    } finally {
      if (latest.current === id) setDetailLoading(false);
    }
  }

  if (session.status === "loading") {
    return <div className="flex h-screen items-center justify-center text-sm text-gray-400">…</div>;
  }

  if (!isAdmin) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-3 text-sm text-gray-500">
        <p>Tài khoản này không có quyền quản trị.</p>
        <Link href="/" className="text-blue-600 hover:text-blue-700">
          ← Quay lại trang chat
        </Link>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col bg-white text-gray-800">
      <header className="flex items-center gap-3 border-b border-gray-200 bg-gray-50 px-4 py-3">
        <Link href="/admin" className="text-xs text-blue-600 hover:text-blue-700">
          ← Khách hàng
        </Link>
        <h1 className="text-sm font-semibold text-gray-700">
          Hội thoại của <span className="font-mono">{customerId}</span>
        </h1>
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

      {error && (
        <div className="border-b border-rose-200 bg-rose-50 px-4 py-2 text-xs text-rose-700">{error}</div>
      )}

      <div className="flex flex-1 overflow-hidden">
        <aside className="flex w-96 flex-col border-r border-gray-200">
          <div className="flex items-center justify-between border-b border-gray-100 px-4 py-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
              Hội thoại ({items.length}
              {cursor ? "+" : ""})
            </span>
            <button
              onClick={() => loadPage(null)}
              disabled={listLoading}
              className="text-xs text-gray-400 hover:text-gray-600 disabled:opacity-40"
            >
              {listLoading ? "…" : "Refresh"}
            </button>
          </div>
          <ul className="flex-1 overflow-y-auto p-2">
            {items.length === 0 && !listLoading && (
              <li className="px-2 py-6 text-center text-xs text-gray-400">
                Khách hàng này chưa có hội thoại nào.
              </li>
            )}
            {items.map((c) => {
              const active = selId === c.conversation_id;
              return (
                <li key={c.conversation_id}>
                  <button
                    onClick={() => open(c.conversation_id)}
                    className={`mb-1 w-full rounded-md border px-3 py-2 text-left transition-colors ${
                      active
                        ? "border-blue-400 bg-blue-50 ring-1 ring-blue-300"
                        : "border-transparent hover:border-gray-200 hover:bg-gray-50"
                    }`}
                  >
                    <p className="line-clamp-2 text-sm text-gray-700">
                      {c.title || <span className="italic text-gray-400">(không có nội dung)</span>}
                    </p>
                    <p className="mt-1 text-[11px] text-gray-400">
                      {formatTime(c.updated_at)} · {c.turn_count} tin nhắn
                    </p>
                  </button>
                </li>
              );
            })}
            {cursor && (
              <li className="px-2 py-2 text-center">
                <button
                  onClick={() => loadPage(cursor)}
                  disabled={listLoading}
                  className="text-xs font-medium text-blue-600 hover:text-blue-700 disabled:opacity-40"
                >
                  {listLoading ? "…" : "Tải thêm"}
                </button>
              </li>
            )}
          </ul>
        </aside>

        <section className="flex flex-1 flex-col overflow-hidden">
          {detail ? (
            <>
              <div className="border-b border-gray-100 px-4 py-2 text-[11px] text-gray-400">
                <span className="font-mono">{detail.conversation_id}</span>
              </div>
              <MessageList messages={toMessages(detail)} loading={false} />
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center text-sm text-gray-400">
              {detailLoading ? "Đang tải…" : "Chọn một hội thoại để xem chi tiết."}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
