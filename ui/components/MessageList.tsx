// ui/components/MessageList.tsx
"use client";

import { useEffect, useRef } from "react";

export interface Message {
  role: "user" | "agent" | "error";
  content: string;
  messageId?: string;
}

interface Props {
  messages: Message[];
  loading: boolean;
  onFeedbackDown?: (messageId: string) => void;
}

export default function MessageList({ messages, loading, onFeedbackDown }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
      {messages.map((msg, i) => (
        <div
          key={i}
          className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
        >
          <div
            className={`max-w-[75%] rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap ${
              msg.role === "user"
                ? "bg-blue-500 text-white"
                : msg.role === "error"
                ? "bg-red-100 text-red-700 border border-red-300"
                : "bg-gray-100 text-gray-800"
            }`}
          >
            {msg.content}
            {msg.role === "agent" && msg.messageId && onFeedbackDown && (
              <button
                aria-label="Không hữu ích"
                onClick={() => onFeedbackDown(msg.messageId!)}
                style={{ marginLeft: 8, opacity: 0.6 }}
              >
                👎
              </button>
            )}
          </div>
        </div>
      ))}

      {loading && (
        <div className="flex justify-start">
          <div className="rounded-2xl bg-gray-100 px-4 py-2">
            <span className="flex gap-1">
              <span className="animate-bounce text-gray-400" style={{ animationDelay: "0ms" }}>●</span>
              <span className="animate-bounce text-gray-400" style={{ animationDelay: "150ms" }}>●</span>
              <span className="animate-bounce text-gray-400" style={{ animationDelay: "300ms" }}>●</span>
            </span>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
