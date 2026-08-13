// ui/components/MessageList.tsx
"use client";

import {
  createContext,
  Fragment,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type MouseEvent,
  type ReactNode,
} from "react";

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
 * A single image line, e.g. `![screen](https://…)`. Nothing else on the line.
 * Used to give a screenshot its own block instead of wedging it into a paragraph.
 */
const IMAGE_LINE = /^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$/;

/**
 * Opens the full-size viewer. A context rather than a prop because the images sit at the
 * bottom of the markdown renderer (`Markdown` → block/`Inline` → `DocImage`), and
 * threading a callback through those would mean touching every intermediate signature.
 */
const LightboxContext = createContext<(url: string) => void>(() => {});

/** Wraps a thumbnail so a plain click opens the viewer instead of navigating.
 *
 * Kept as a real `<a href>` rather than a button: modifier- and middle-clicks then still
 * open the image in a new tab through the browser's own handling, which is worth
 * preserving for anyone who wants the raw file. Only an unmodified left click is
 * intercepted.
 */
function ImageTrigger({
  url,
  title,
  className,
  children,
}: {
  url: string;
  title: string;
  className?: string;
  children: ReactNode;
}) {
  const open = useContext(LightboxContext);
  const onClick = (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
    e.preventDefault();
    open(url);
  };
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      onClick={onClick}
      title={title}
      className={className}
    >
      {children}
    </a>
  );
}

/**
 * Full-size image viewer. Dismissed by the close button, a click on the backdrop, or
 * Escape — all three, because a modal that traps the reader in a chat window is worse
 * than no modal. Body scroll is locked while open so the page behind stays put.
 */
function Lightbox({ url, onClose }: { url: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Ảnh phóng to"
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 sm:p-8"
    >
      <button
        type="button"
        onClick={onClose}
        aria-label="Đóng"
        title="Đóng (Esc)"
        className="absolute right-3 top-3 flex h-9 w-9 items-center justify-center rounded-full bg-white/15 text-xl leading-none text-white transition-colors hover:bg-white/30"
      >
        ×
      </button>
      <img
        src={url}
        alt="Ảnh phóng to"
        // Stop the backdrop handler: clicking the image itself should not dismiss it.
        onClick={(e) => e.stopPropagation()}
        className="max-h-full max-w-full rounded-lg bg-white object-contain shadow-2xl"
      />
    </div>
  );
}

/**
 * Images extracted from the source user guides, sent by the server as markdown with a
 * presigned URL. The alt text carries the *kind*, which is the only channel that
 * survives into rendered markdown, and it decides the treatment:
 *
 *  - `icon`   — a button glyph from a guide's table. Renders at glyph size inline in
 *               the sentence, so "Nhấn [icon] để tạo hồ sơ" reads as one instruction.
 *  - `screen` — a full screenshot. Renders as a bounded preview that opens the full
 *               image in the lightbox, matching how user attachments behave below.
 */
function DocImage({ kind, url }: { kind: string; url: string }) {
  if (kind === "icon") {
    return (
      <img
        src={url}
        alt="Biểu tượng trên phần mềm"
        className="inline-block max-h-5 max-w-[8rem] align-text-bottom rounded-sm border border-gray-300 bg-white object-contain"
      />
    );
  }
  return (
    <ImageTrigger
      url={url}
      title="Nhấn để xem ảnh đầy đủ"
      className="group mt-1 inline-flex cursor-zoom-in flex-col gap-0.5"
    >
      <img
        src={url}
        alt="Ảnh minh hoạ từ tài liệu hướng dẫn"
        className="max-h-40 max-w-full rounded-lg border border-gray-300 bg-white object-contain transition-colors group-hover:border-blue-400"
      />
      <span className="text-[0.7rem] text-gray-500 group-hover:text-blue-600 group-hover:underline">
        Nhấn để xem ảnh đầy đủ
      </span>
    </ImageTrigger>
  );
}

/**
 * Inline markdown within a single line: `![alt](url)`, `**bold**` and `` `code` ``.
 * Anything else is passed through untouched. We tokenize with one regex so the
 * markers can appear in any order without nesting bugs — the image alternative
 * comes first so `![…](…)` is never mistaken for other syntax.
 */
function Inline({ text }: { text: string }) {
  const nodes: ReactNode[] = [];
  const regex = /!\[([^\]]*)\]\(([^)]+)\)|\*\*(.+?)\*\*|`([^`]+)`/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[2] !== undefined) {
      nodes.push(<DocImage key={key++} kind={m[1]} url={m[2]} />);
    } else if (m[3] !== undefined) {
      nodes.push(
        <strong key={key++} className="font-semibold">
          {m[3]}
        </strong>,
      );
    } else {
      nodes.push(
        <code key={key++} className="rounded bg-black/5 px-1 py-0.5 text-[0.85em]">
          {m[4]}
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
  | { type: "p"; lines: string[] }
  | { type: "img"; alt: string; url: string };

/**
 * Group raw text into markdown blocks line-by-line: consecutive `- `/`* ` lines
 * become a bullet list, `1.` lines an ordered list, a line holding only an image
 * becomes its own block, blank lines break paragraphs, everything else
 * accumulates into a paragraph.
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
    const img = line.match(IMAGE_LINE);
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*\d+\.\s+(.*)$/);
    if (img) {
      // A standalone screenshot gets real block margins rather than being absorbed
      // into whatever paragraph or list happened to precede it. Checked before the
      // list patterns because `![…](…)` alone on a line is never a list item.
      flushPara();
      flushList();
      blocks.push({ type: "img", alt: img[1], url: img[2] });
    } else if (ul) {
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
 * Assign each ordered list a `start` so numbering continues across an interruption.
 *
 * `parseBlocks` ends a list at any blank line, sub-list, or image, so a reply whose steps
 * read "1. 2. 3." in markdown became three separate `<ol>`s each restarting at 1. What
 * breaks a sequence is a *paragraph* — "**Thao tác nhận mẫu:**" genuinely starts a new
 * list — while an illustrating screenshot or a nested bullet list belongs to the step it
 * follows and must not reset the count.
 *
 * This is a numbering fix, not real nesting: sub-bullets still render as a sibling list
 * rather than inside their `<li>`. Correcting that means teaching `parseBlocks` about
 * indentation, which is more than the chat replies need.
 */
function withOrderedStarts(blocks: Block[]): (Block & { start?: number })[] {
  let next = 1;
  let open = false;
  return blocks.map((b) => {
    if (b.type === "ol") {
      const start = open ? next : 1;
      next = start + b.items.length;
      open = true;
      return { ...b, start };
    }
    if (b.type === "p") {
      open = false;
      next = 1;
    }
    return b;
  });
}

/**
 * Lightweight markdown renderer covering what the chat responses use:
 * paragraphs, bullet/ordered lists, images, inline bold and code. Avoids a full
 * markdown dependency; the container no longer needs `whitespace-pre-wrap`
 * since block structure is now explicit.
 */
function Markdown({ text }: { text: string }) {
  const blocks = withOrderedStarts(parseBlocks(text));
  return (
    <div className="space-y-2">
      {blocks.map((b, i) => {
        if (b.type === "img") {
          return (
            <div key={i}>
              <DocImage kind={b.alt} url={b.url} />
            </div>
          );
        }
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
          <ListTag key={i} className={`${listClass} space-y-1 pl-5`} start={b.start}>
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
  // URL of the image being viewed full size, or null when the viewer is closed. Held here
  // rather than per-image so only one can ever be open.
  const [zoomed, setZoomed] = useState<string | null>(null);
  // Stable identity: it is the context value, so a new function each render would
  // re-render every image in the thread.
  const closeZoom = useCallback(() => setZoomed(null), []);

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
    <LightboxContext.Provider value={setZoomed}>
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
                        // Same lightbox as the guide screenshots — two different
                        // click behaviours for two kinds of image in one thread
                        // would just read as a bug.
                        <ImageTrigger
                          key={j}
                          url={att.url}
                          title="Nhấn để xem ảnh đầy đủ"
                          className="cursor-zoom-in"
                        >
                          <img
                            src={att.url}
                            alt={`Ảnh đính kèm ${j + 1}`}
                            className="max-h-48 max-w-full rounded-lg border border-blue-300 object-contain"
                          />
                        </ImageTrigger>
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
                  <div className="rounded-2xl rounded-tl-sm px-4 py-2 text-sm bg-gray-100 text-gray-800">
                    <Markdown text={msg.content} />
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
            <div className="max-w-[75%] rounded-2xl px-4 py-2 text-sm bg-red-100 text-red-700 border border-red-300">
              <Markdown text={msg.content} />
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
      {zoomed && <Lightbox url={zoomed} onClose={closeZoom} />}
    </LightboxContext.Provider>
  );
}
