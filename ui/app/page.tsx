"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import ConfigBar, { needsApplicationChoice } from "@/components/ConfigBar";
import MessageList, { Message } from "@/components/MessageList";
import InputBar from "@/components/InputBar";
import { sendMessage, sendFeedback, Attachment, UnauthorizedError } from "@/lib/api";
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
      setMessages((prev) => {
        const next = [...prev];
        // The uploaded images belong to the user message we optimistically added
        // above; the server only knows their presigned URLs now that it has stored
        // them. Patch that message in place so the chips become real thumbnails.
        if (result.attachments?.length) {
          const i = next.findLastIndex((m) => m.role === "user");
          if (i !== -1) next[i] = { ...next[i], attachments: result.attachments };
        }
        return [...next, { role: "agent", content: result.reply, messageId: result.message_id }];
      });
    } catch (err) {
      if (err instanceof UnauthorizedError) {
        logout(router);
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
        onConversationIdChange={setConversationId}
        onApplicationsChange={setSelectedApplications}
        onNewConversation={handleNewConversation}
        onLogout={() => logout(router)}
      />
      <MessageList messages={messages} loading={loading} onFeedbackDown={handleFeedbackDown} />
      <InputBar
        onSend={handleSend}
        disabled={loading}
        blockedReason={
          mustChooseApplication ? "Chọn ứng dụng ở trên trước khi đặt câu hỏi…" : undefined
        }
      />
    </div>
  );
}
