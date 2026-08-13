// ui/components/MessageList.tsx
"use client";

import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";

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

/**
 * Inline markdown within a single line: `**bold**` and `` `code` ``. Anything
 * else is passed through untouched. We tokenize with one regex so the two
 * markers can appear in any order without nesting bugs.
 */
function Inline({ text }: { text: string }) {
  const nodes: ReactNode[] = [];
  const regex = /\*\*(.+?)\*\*|`([^`]+)`/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[1] !== undefined) {
      nodes.push(
        <strong key={key++} className="font-semibold">
          {m[1]}
        </strong>,
      );
    } else {
      nodes.push(
        <code key={key++} className="rounded bg-black/5 px-1 py-0.5 text-[0.85em]">
          {m[2]}
        </code>,
      );
    }
    last = regex.lastIndex;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return <>{nodes}</>;
}

type Block =
  | { type: "ul" | "ol"; items: string[] }
  | { type: "p"; lines: string[] };

/**
 * Group raw text into markdown blocks line-by-line: consecutive `- `/`* ` lines
 * become a bullet list, `1.` lines an ordered list, blank lines break
 * paragraphs, everything else accumulates into a paragraph.
 */
function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: { type: "ul" | "ol"; items: string[] } | null = null;

  const flushPara = () => {
    if (para.length) {
      blocks.push({ type: "p", lines: para });
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      blocks.push(list);
      list = null;
    }
  };

  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*\d+\.\s+(.*)$/);
    if (ul) {
      flushPara();
      if (!list || list.type !== "ul") {
        flushList();
        list = { type: "ul", items: [] };
      }
      list.items.push(ul[1]);
    } else if (ol) {
      flushPara();
      if (!list || list.type !== "ol") {
        flushList();
        list = { type: "ol", items: [] };
      }
      list.items.push(ol[1]);
    } else if (line.trim() === "") {
      flushPara();
      flushList();
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return blocks;
}

/**
 * Lightweight markdown renderer covering what the chat responses use:
 * paragraphs, bullet/ordered lists, inline bold and code. Avoids a full
 * markdown dependency; the container no longer needs `whitespace-pre-wrap`
 * since block structure is now explicit.
 */
function Markdown({ text }: { text: string }) {
  const blocks = parseBlocks(text);
  return (
    <div className="space-y-2">
      {blocks.map((b, i) => {
        if (b.type === "p") {
          return (
            <p key={i}>
              {b.lines.map((ln, j) => (
                <Fragment key={j}>
                  {j > 0 && <br />}
                  <Inline text={ln} />
                </Fragment>
              ))}
            </p>
          );
        }
        const ListTag = b.type === "ul" ? "ul" : "ol";
        const listClass = b.type === "ul" ? "list-disc" : "list-decimal";
        return (
          <ListTag key={i} className={`${listClass} space-y-1 pl-5`}>
            {b.items.map((it, j) => (
              <li key={j}>
                <Inline text={it} />
              </li>
            ))}
          </ListTag>
        );
      })}
    </div>
  );
}

/**
 * The agent's avatar. A plain `<img>` rather than `next/image`: it is a fixed 24px
 * square served from `public/`, so the optimizer has nothing to do and would only
 * add a request. `cenlab-mark.png` is the logo cropped to the graphic — the full
 * lockup carries a wordmark that is unreadable at this size.
 */
function AgentAvatar() {
  return (
    // eslint-disable-next-line @next/next/no-img-element -- fixed 24px static asset; the optimizer has nothing to do here.
    <img
      src="/cenlab-mark.png"
      alt="CenLab"
      className="mt-1 h-6 w-6 shrink-0 select-none rounded-full border border-gray-200 bg-white object-contain p-0.5"
    />
  );
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

/** Two overlapping sheets for "copy", swapped for a tick once the copy lands. */
function CopyIcon({ done }: { done: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-4 w-4"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {done ? (
        <path d="m20 6-11 11-5-5" />
      ) : (
        <>
          <rect x="9" y="9" width="12" height="12" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </>
      )}
    </svg>
  );
}

export default function MessageList({ messages, loading, onFeedbackDown }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  // Per-message selection, keyed by messageId. undefined = nothing picked yet.
  const [feedback, setFeedback] = useState<Record<string, FeedbackSignal | undefined>>({});
  // Id of the message whose text was just copied, so only that row shows the tick.
  const [copied, setCopied] = useState<string | null>(null);
  const copyTimer = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => () => clearTimeout(copyTimer.current), []);

  /**
   * `navigator.clipboard` needs a secure context — fine on localhost and https,
   * absent over plain http. Failing silently is deliberate: the button is a
   * convenience, and the message text is on screen either way.
   */
  const handleCopy = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      return;
    }
    setCopied(messageId);
    clearTimeout(copyTimer.current);
    copyTimer.current = setTimeout(() => setCopied(null), 1500);
  };

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
                <AgentAvatar />
                <div className="flex flex-col gap-0.5">
                  <div className="rounded-2xl rounded-tl-sm px-4 py-2 text-sm bg-gray-100 text-gray-800">
                    <Markdown text={msg.content} />
                  </div>
                  {msg.messageId && (
                    <div className="flex items-center gap-1 pl-1 h-7">
                      <button
                        type="button"
                        title="Hữu ích"
                        aria-pressed={feedback[msg.messageId] === "up"}
                        onClick={() => handleFeedback(msg.messageId!, "up")}
                        className={`flex items-center gap-1 rounded-md px-1.5 py-1 text-xs transition-colors ${
                          feedback[msg.messageId] === "up"
                            ? "text-blue-600 bg-blue-50"
                            : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                        }`}
                      >
                        <ThumbIcon filled={feedback[msg.messageId] === "up"} />
                        Hữu ích
                      </button>
                      <button
                        type="button"
                        title="Chưa đúng"
                        aria-pressed={feedback[msg.messageId] === "down"}
                        onClick={() => handleFeedback(msg.messageId!, "down")}
                        className={`flex items-center gap-1 rounded-md px-1.5 py-1 text-xs transition-colors ${
                          feedback[msg.messageId] === "down"
                            ? "text-red-500 bg-red-50"
                            : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                        }`}
                      >
                        {/* Only the glyph flips; rotating the label would mirror the text. */}
                        <span className="rotate-180">
                          <ThumbIcon filled={feedback[msg.messageId] === "down"} />
                        </span>
                        Chưa đúng
                      </button>
                      <button
                        type="button"
                        title="Sao chép câu trả lời"
                        onClick={() => handleCopy(msg.messageId!, msg.content)}
                        className={`flex items-center gap-1 rounded-md px-1.5 py-1 text-xs transition-colors ${
                          copied === msg.messageId
                            ? "text-emerald-600 bg-emerald-50"
                            : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                        }`}
                      >
                        <CopyIcon done={copied === msg.messageId} />
                        {copied === msg.messageId ? "Đã chép" : "Sao chép"}
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
            <div className="max-w-[75%] rounded-2xl px-4 py-2 text-sm bg-red-100 text-red-700 border border-red-300">
              <Markdown text={msg.content} />
            </div>
          </div>
        );
      })}

      {loading && (
        <div className="flex justify-start">
          <div className="flex gap-2">
            <AgentAvatar />
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
