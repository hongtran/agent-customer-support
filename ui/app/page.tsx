"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ConfigBar, { needsApplicationChoice } from "@/components/ConfigBar";
import MessageList, { Message } from "@/components/MessageList";
import InputBar from "@/components/InputBar";
import {
  sendMessage,
  sendFeedback,
  Attachment,
  RateLimitError,
  UnauthorizedError,
} from "@/lib/api";
import { logout, useSession } from "@/lib/useSession";

/**
 * A conversation id for a brand-new thread.
 *
 * `crypto.randomUUID` only exists in a secure context (https, or localhost), so a
 * demo served over plain http on a LAN address would not have it. That used to
 * break one button; now it would break the first render, so fall back rather than
 * throw. The fallback is not cryptographically strong and does not need to be —
 * the id only has to be unique per browser tab, and the server reads the tenant
 * from the access token, never from this string.
 */
function newConversationId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `conv-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export default function Home() {
  const router = useRouter();
  const session = useSession();
  // Every visit to the chat page starts its own conversation. The old hardcoded
  // "smoke-ui" default meant every page load appended to one shared transcript, so
  // the agent re-read a stranger's history as context. A lazy initializer, so the id
  // is minted once per mount instead of on every render.
  const [conversationId, setConversationId] = useState(newConversationId);
  const [selectedApplications, setSelectedApplications] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  // Questions left today; null = unlimited. Seeded from /auth/me, then replaced by the
  // fresh value each chat response carries, so the header never needs a separate poll.
  const [remaining, setRemaining] = useState<number | null>(null);
  const limitReached = remaining === 0;
  // The centered notice can be closed so earlier answers stay readable; it comes back
  // if the user hits the limit again (e.g. a question refused with 429).
  const [limitNoticeClosed, setLimitNoticeClosed] = useState(false);

  useEffect(() => {
    if (session.status === "ready") setRemaining(session.me.questions_remaining);
  }, [session]);

  const handleFeedbackDown = (messageId: string) => {
    sendFeedback(conversationId, messageId).catch(() => {});
  };

  const handleNewConversation = () => {
    setConversationId(newConversationId());
    setMessages([]);
    setSelectedApplications([]);
  };

  const handleSend = async (text: string, attachments: Attachment[]) => {
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: text,
        attachments:
          attachments.length > 0 ? attachments.map((a) => ({ media_type: a.media_type })) : undefined,
      },
    ]);
    setLoading(true);
    try {
      const result = await sendMessage({
        // customer_id is gone: the server reads the tenant from the access token.
        conversation_id: conversationId,
        message: text,
        attachments: attachments.length > 0 ? attachments : undefined,
        applications: selectedApplications.length > 0 ? selectedApplications : undefined,
      });
      setRemaining(result.questions_remaining ?? null);
      setMessages((prev) => {
        const next = [...prev];
        // The uploaded images belong to the user message we optimistically added
        // above; the server only knows their presigned URLs now that it has stored
        // them. Patch that message in place so the chips become real thumbnails.
        if (result.attachments?.length) {
          const i = next.findLastIndex((m) => m.role === "user");
          if (i !== -1) next[i] = { ...next[i], attachments: result.attachments };
        }
        return [
          ...next,
          {
            role: "agent",
            content: result.reply,
            messageId: result.message_id,
            citations: result.citations,
          },
        ];
      });
    } catch (err) {
      if (err instanceof UnauthorizedError) {
        logout(router);
        return;
      }
      if (err instanceof RateLimitError) {
        setRemaining(0);
        setLimitNoticeClosed(false);
        setMessages((prev) => [...prev, { role: "warning", content: err.message }]);
        return;
      }
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [...prev, { role: "error", content: `Error: ${msg}` }]);
    } finally {
      setLoading(false);
    }
  };

  if (session.status === "loading") {
    return <div className="flex h-screen items-center justify-center text-sm text-gray-400">…</div>;
  }

  // The picker in ConfigBar owns this rule; the composer only mirrors it, so both
  // read it from the same place instead of each deciding what "not chosen" means.
  const mustChooseApplication = needsApplicationChoice(
    session.me.enabled_applications,
    selectedApplications
  );

  return (
    <div className="flex h-screen flex-col">
      <ConfigBar
        me={session.me}
        conversationId={conversationId}
        selectedApplications={selectedApplications}
        questionsRemaining={remaining}
        onConversationIdChange={setConversationId}
        onApplicationsChange={setSelectedApplications}
        onNewConversation={handleNewConversation}
        onLogout={() => logout(router)}
      />
      <div className="relative flex flex-1 flex-col overflow-hidden">
        <MessageList messages={messages} loading={loading} onFeedbackDown={handleFeedbackDown} />
        {limitReached && !limitNoticeClosed && (
          // pointer-events-none on the layer, so the chat behind stays scrollable;
          // only the card itself takes clicks.
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-4">
            <div
              role="alert"
              className="pointer-events-auto max-w-md rounded-2xl border border-amber-300 bg-amber-50 px-6 py-5 text-center shadow-lg"
            >
              <div className="text-3xl" aria-hidden="true">
                ⚠️
              </div>
              <p className="mt-2 text-base font-medium text-amber-900">
                Anh/Chị đã hết lượt hỏi hôm nay. Vui lòng quay lại vào ngày mai.
              </p>
              <button
                onClick={() => setLimitNoticeClosed(true)}
                className="mt-4 rounded-md border border-amber-300 bg-white px-4 py-1.5 text-sm text-amber-800 hover:bg-amber-100"
              >
                Đóng
              </button>
            </div>
          </div>
        )}
      </div>
      <InputBar
        onSend={handleSend}
        disabled={loading}
        blockedReason={
          limitReached
            ? "Anh/Chị đã hết lượt hỏi hôm nay. Vui lòng quay lại vào ngày mai."
            : mustChooseApplication
              ? "Chọn ứng dụng ở trên trước khi đặt câu hỏi…"
              : undefined
        }
      />
    </div>
  );
}
