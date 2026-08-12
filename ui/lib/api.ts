// ui/lib/api.ts

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8800";

const TOKEN_KEY = "accessToken";
const ROLE_KEY = "accessRole";

/* ---------------------------------------------------------------- session */

export type Role = "admin" | "user";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function getRole(): Role | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ROLE_KEY) as Role | null;
}

export function clearSession() {
  if (typeof window === "undefined") return;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(ROLE_KEY);
}

/** Thrown on a 401 so callers can bounce to /login instead of showing a raw error. */
export class UnauthorizedError extends Error {
  constructor() {
    super("Session expired");
    this.name = "UnauthorizedError";
  }
}

function authHeaders(json = true): Record<string, string> {
  const headers: Record<string, string> = {};
  if (json) headers["Content-Type"] = "application/json";
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

/**
 * Single place every authenticated call goes through. A 401 clears the stored token
 * before throwing — leaving a dead token in localStorage means every later call fails
 * silently and the user never gets sent back to the login screen.
 */
async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { ...authHeaders(), ...(init.headers as Record<string, string>) },
  });
  if (res.status === 401) {
    clearSession();
    throw new UnauthorizedError();
  }
  return res;
}

/* ------------------------------------------------------------------- auth */

export interface LoginResult {
  access_token: string;
  token_type: string;
  customer_id: string;
  role: Role;
  expires_in: number;
}

export async function login(userName: string, password: string): Promise<LoginResult> {
  const res = await fetch(`${BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_name: userName, password }),
  });
  if (res.status === 401) throw new Error("Sai tên đăng nhập hoặc mật khẩu");
  if (!res.ok) throw new Error(`Đăng nhập thất bại (${res.status})`);
  const data: LoginResult = await res.json();
  localStorage.setItem(TOKEN_KEY, data.access_token);
  localStorage.setItem(ROLE_KEY, data.role);
  return data;
}

export interface Me {
  customer_id: string;
  name: string;
  role: Role;
  enabled_applications: string[];
}

export async function getMe(): Promise<Me> {
  const res = await request("/auth/me");
  if (!res.ok) throw new Error(`me failed: ${res.status}`);
  return res.json();
}

/* ----------------------------------------------------------------- widget */

export interface Attachment {
  kind: "image";
  media_type: string;
  data: string; // base64, no data: prefix
}

export interface ChatPayload {
  // No customer_id: the server takes the tenant from the access token.
  conversation_id: string;
  message: string;
  attachments?: Attachment[];
  applications?: string[];
}

/** An uploaded image as the server hands it back: a presigned, expiring S3 URL. */
export interface AttachmentRef {
  kind: "image";
  media_type: string;
  url: string;
}

export interface ChatResult {
  reply: string;
  message_id?: string;
  /** The images from the message just sent, now stored and signed for display. */
  attachments?: AttachmentRef[];
}

export async function sendMessage(payload: ChatPayload): Promise<ChatResult> {
  const res = await request("/widget/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`Server error ${res.status}: ${await res.text()}`);
  }

  const data = await res.json();
  return { reply: data.reply, message_id: data.message_id, attachments: data.attachments };
}

export async function getMyApplications(): Promise<string[]> {
  const res = await request("/widget/me/applications");
  if (!res.ok) return [];
  const data = await res.json();
  return data.enabled_applications ?? [];
}

export async function sendFeedback(conversationId: string, messageId: string) {
  await request("/widget/feedback", {
    method: "POST",
    body: JSON.stringify({ conversation_id: conversationId, message_id: messageId, signal: "down" }),
  });
}

/* ------------------------------------------------------------------ admin */

export interface Application {
  name: string;
  slug: string;
}

/** The canonical application catalogue, used to populate admin dropdowns. */
export async function listApplications(): Promise<Application[]> {
  const r = await request("/admin/applications");
  if (!r.ok) throw new Error(`list applications failed: ${r.status}`);
  return r.json();
}

export async function listQA(status = "pending") {
  const r = await request(`/admin/qa?status=${status}`);
  if (!r.ok) throw new Error(`list failed: ${r.status}`);
  return r.json();
}

export async function createQA(body: {
  question: string;
  answer?: string;
  application?: string | null;
}) {
  const r = await request("/admin/qa", { method: "POST", body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`create failed: ${r.status}`);
  return r.json();
}

export async function approveQA(id: string) {
  const r = await request(`/admin/qa/${id}/approve`, { method: "POST", body: "{}" });
  if (!r.ok) throw new Error(`approve failed: ${r.status}`);
  return r.json();
}

export async function rejectQA(id: string) {
  const r = await request(`/admin/qa/${id}/reject`, { method: "POST", body: "{}" });
  if (!r.ok) throw new Error(`reject failed: ${r.status}`);
  return r.json();
}

export async function editQA(id: string, patch: Record<string, unknown>) {
  const r = await request(`/admin/qa/${id}`, { method: "PATCH", body: JSON.stringify(patch) });
  if (!r.ok) throw new Error(`edit failed: ${r.status}`);
  return r.json();
}

/* -------------------------------------------------- admin: customers */

export interface Customer {
  customer_id: string;
  name: string;
  role: Role;
  enabled_applications: string[];
  config_notes: string | null;
  has_password: boolean;
}

export interface CustomerCreate {
  customer_id: string;
  name: string;
  password: string;
  role?: Role;
  enabled_applications?: string[];
  config_notes?: string | null;
}

export type CustomerPatch = Partial<Omit<CustomerCreate, "customer_id">>;

export async function listCustomers(): Promise<Customer[]> {
  const r = await request("/admin/customers");
  if (!r.ok) throw new Error(`list customers failed: ${r.status}`);
  return r.json();
}

export async function createCustomer(body: CustomerCreate): Promise<Customer> {
  const r = await request("/admin/customers", { method: "POST", body: JSON.stringify(body) });
  if (r.status === 409) throw new Error(`Tên đăng nhập "${body.customer_id}" đã tồn tại`);
  if (!r.ok) throw new Error(`create customer failed: ${r.status}`);
  return r.json();
}

export async function updateCustomer(id: string, patch: CustomerPatch): Promise<Customer> {
  const r = await request(`/admin/customers/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
  if (!r.ok) throw new Error(`update customer failed: ${r.status}`);
  return r.json();
}
