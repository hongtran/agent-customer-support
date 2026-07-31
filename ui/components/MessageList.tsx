// ui/components/MessageList.tsx
"use client";

import { useEffect, useRef, useState } from "react";

export interface Message {
  role: "user" | "agent" | "error";
  content: string;
  messageId?: string;
  // `url` is a presigned S3 link, filled in from the chat response. It is absent
  // for the optimistic echo rendered before the request completes, so the chip
  // below doubles as the pre-upload placeholder.
  attachments?: { media_type: string; url?: string }[];
}

type FeedbackSignal = "up" | "down";

interface Props {
  messages: Message[];
  loading: boolean;
  onFeedbackDown?: (messageId: string) => void;
}

/** Thumbs-up outline; the dislike button reuses it rotated 180°. */
function ThumbIcon({ filled }: { filled: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="w-4 h-4"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M7 10v12" />
      <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
    </svg>
  );
}

export default function MessageList({ messages, loading, onFeedbackDown }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  // Per-message selection, keyed by messageId. undefined = nothing picked yet.
  const [feedback, setFeedback] = useState<Record<string, FeedbackSignal | undefined>>({});

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleFeedback = (messageId: string, signal: FeedbackSignal) => {
    const next = feedback[messageId] === signal ? undefined : signal; // click again to clear
    setFeedback((prev) => ({ ...prev, [messageId]: next }));
    // Only a dislike reaches the backend — likes are UI-only for now.
    if (next === "down") onFeedbackDown?.(messageId);
  };

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
      {messages.map((msg, i) => {
        if (msg.role === "user") {
          return (
            <div key={i} className="flex justify-end">
              <div className="flex flex-col items-end gap-1 max-w-[75%]">
                <div className="rounded-2xl rounded-tr-sm px-4 py-2 text-sm whitespace-pre-wrap bg-blue-500 text-white">
                  {msg.content}
                </div>
                {msg.attachments && msg.attachments.length > 0 && (
                  <div className="flex flex-wrap gap-1 justify-end">
                    {msg.attachments.map((att, j) =>
                      att.url ? (
                        <a key={j} href={att.url} target="_blank" rel="noopener noreferrer">
                          <img
                            src={att.url}
                            alt={`Ảnh đính kèm ${j + 1}`}
                            className="max-h-48 max-w-full rounded-lg border border-blue-300 object-contain"
                          />
                        </a>
                      ) : (
                        <span
                          key={j}
                          className="flex items-center gap-1 rounded-lg border border-blue-300 bg-blue-50 text-blue-700 px-2 py-0.5 text-xs"
                        >
                          🖼 {att.media_type.replace("image/", "")}
                        </span>
                      ),
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        }

        if (msg.role === "agent") {
          return (
            <div key={i} className="flex justify-start">
              <div className="flex gap-2 max-w-[80%]">
                <div className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center text-xs shrink-0 mt-1 select-none">
                  🤖
                </div>
                <div className="flex flex-col gap-0.5">
                  <div className="rounded-2xl rounded-tl-sm px-4 py-2 text-sm whitespace-pre-wrap bg-gray-100 text-gray-800">
                    {msg.content}
                  </div>
                  {msg.messageId && (
                    <div className="flex items-center gap-1 pl-1 h-7">
                      <button
                        type="button"
                        aria-label="Hữu ích"
                        title="Hữu ích"
                        aria-pressed={feedback[msg.messageId] === "up"}
                        onClick={() => handleFeedback(msg.messageId!, "up")}
                        className={`rounded-md p-1 transition-colors ${
                          feedback[msg.messageId] === "up"
                            ? "text-blue-600 bg-blue-50"
                            : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                        }`}
                      >
                        <ThumbIcon filled={feedback[msg.messageId] === "up"} />
                      </button>
                      <button
                        type="button"
                        aria-label="Không hữu ích"
                        title="Không hữu ích"
                        aria-pressed={feedback[msg.messageId] === "down"}
                        onClick={() => handleFeedback(msg.messageId!, "down")}
                        className={`rounded-md p-1 rotate-180 transition-colors ${
                          feedback[msg.messageId] === "down"
                            ? "text-red-500 bg-red-50"
                            : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                        }`}
                      >
                        <ThumbIcon filled={feedback[msg.messageId] === "down"} />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        }

        return (
          <div key={i} className="flex justify-start">
            <div className="max-w-[75%] rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap bg-red-100 text-red-700 border border-red-300">
              {msg.content}
            </div>
          </div>
        );
      })}

      {loading && (
        <div className="flex justify-start">
          <div className="flex gap-2">
            <div className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center text-xs shrink-0 mt-1 select-none">
              🤖
            </div>
            <div className="rounded-2xl rounded-tl-sm bg-gray-100 px-4 py-2">
              <span className="flex gap-1">
                <span className="animate-bounce text-gray-400" style={{ animationDelay: "0ms" }}>●</span>
                <span className="animate-bounce text-gray-400" style={{ animationDelay: "150ms" }}>●</span>
                <span className="animate-bounce text-gray-400" style={{ animationDelay: "300ms" }}>●</span>
              </span>
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
