// ui/lib/api.ts

const BASE = "http://localhost:8000";

export interface ChatPayload {
  customer_id: string;
  conversation_id: string;
  message: string;
}

export interface ChatResult {
  reply: string;
}

export async function sendMessage(payload: ChatPayload): Promise<ChatResult> {
  const res = await fetch(`${BASE}/widget/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`Server error ${res.status}: ${await res.text()}`);
  }

  const data = await res.json();
  return { reply: data.reply };
}
