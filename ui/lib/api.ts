// ui/lib/api.ts

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8800";

export interface Attachment {
  kind: "image";
  media_type: string;
  data: string; // base64, no data: prefix
}

export interface ChatPayload {
  customer_id: string;
  conversation_id: string;
  message: string;
  attachments?: Attachment[];
  applications?: string[];
}

export interface ChatResult {
  reply: string;
  message_id?: string;
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
  return { reply: data.reply, message_id: data.message_id };
}

export async function getCustomerApplications(customerId: string): Promise<string[]> {
  const res = await fetch(`${BASE}/widget/customer/${encodeURIComponent(customerId)}`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.enabled_applications ?? [];
}

export async function sendFeedback(conversationId: string, messageId: string) {
  await fetch(`${BASE}/widget/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, message_id: messageId, signal: "down" }),
  });
}

function adminHeaders(token: string) {
  return { "Content-Type": "application/json", "X-Admin-Token": token };
}

export async function listQA(token: string, status = "pending") {
  const r = await fetch(`${BASE}/admin/qa?status=${status}`, { headers: adminHeaders(token) });
  if (!r.ok) throw new Error(`list failed: ${r.status}`);
  return r.json();
}

export async function createQA(
  token: string,
  body: { question: string; answer?: string; application?: string | null },
) {
  const r = await fetch(`${BASE}/admin/qa`, {
    method: "POST", headers: adminHeaders(token), body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`create failed: ${r.status}`);
  return r.json();
}

export async function approveQA(token: string, id: string) {
  const r = await fetch(`${BASE}/admin/qa/${id}/approve`, {
    method: "POST", headers: adminHeaders(token), body: "{}",
  });
  if (!r.ok) throw new Error(`approve failed: ${r.status}`);
  return r.json();
}

export async function rejectQA(token: string, id: string) {
  const r = await fetch(`${BASE}/admin/qa/${id}/reject`, {
    method: "POST", headers: adminHeaders(token), body: "{}",
  });
  if (!r.ok) throw new Error(`reject failed: ${r.status}`);
  return r.json();
}

export async function editQA(token: string, id: string, patch: Record<string, unknown>) {
  const r = await fetch(`${BASE}/admin/qa/${id}`, {
    method: "PATCH", headers: adminHeaders(token), body: JSON.stringify(patch),
  });
  if (!r.ok) throw new Error(`edit failed: ${r.status}`);
  return r.json();
}
