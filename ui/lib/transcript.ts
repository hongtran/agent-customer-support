import type { Message } from "@/components/MessageList";

const SPEAKER = /^(user|assistant): ?/;

/**
 * A stored transcript (`user: …` / `assistant: …` lines, as the backend writes it) →
 * the chat view's message shape. A reply spans several lines, so a line without a
 * speaker prefix continues the message before it.
 *
 * No `messageId`, so MessageList shows no vote/copy buttons: this is a read-only record.
 */
export function parseTranscript(text: string): Message[] {
  const out: Message[] = [];
  for (const line of text.split("\n")) {
    const m = SPEAKER.exec(line);
    if (m) {
      out.push({ role: m[1] === "user" ? "user" : "agent", content: line.slice(m[0].length) });
    } else if (out.length > 0) {
      out[out.length - 1].content += `\n${line}`;
    } else if (line.trim()) {
      // Text before any speaker line; keep it rather than drop it.
      out.push({ role: "agent", content: line });
    }
  }
  return out;
}
