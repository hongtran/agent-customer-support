"use client";

import { useEffect, useState } from "react";
import { listQA, approveQA, rejectQA, editQA } from "@/lib/api";

type QA = {
  id: string; question: string; answer: string; status: string;
  source: string; bad_answer?: string | null; transcript?: string;
  application?: string | null;
};

export default function AdminPage() {
  const [token, setToken] = useState("");
  const [items, setItems] = useState<QA[]>([]);
  const [sel, setSel] = useState<QA | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const t = localStorage.getItem("adminToken") || "";
    setToken(t);
  }, []);

  async function refresh(t: string) {
    try {
      setError("");
      setItems(await listQA(t, "pending"));
    } catch (e) {
      setError(String(e));
    }
  }

  function saveToken() {
    localStorage.setItem("adminToken", token);
    refresh(token);
  }

  async function onApprove() {
    if (!sel) return;
    await editQA(token, sel.id, { answer: sel.answer, application: sel.application ?? null });
    await approveQA(token, sel.id);
    setSel(null);
    refresh(token);
  }

  async function onReject() {
    if (!sel) return;
    await rejectQA(token, sel.id);
    setSel(null);
    refresh(token);
  }

  return (
    <main style={{ display: "flex", gap: 24, padding: 24 }}>
      <section style={{ width: 360 }}>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          <input value={token} onChange={(e) => setToken(e.target.value)} placeholder="Admin token"
            style={{ flex: 1 }} />
          <button onClick={saveToken}>Load</button>
        </div>
        {error && <p style={{ color: "red" }}>{error}</p>}
        <h3>Pending ({items.length})</h3>
        <ul style={{ listStyle: "none", padding: 0 }}>
          {items.map((it) => (
            <li key={it.id}>
              <button onClick={() => setSel(it)} style={{ textAlign: "left", width: "100%" }}>
                [{it.source}] {it.question}
              </button>
            </li>
          ))}
        </ul>
      </section>
      <section style={{ flex: 1 }}>
        {sel ? (
          <div>
            <h3>{sel.question}</h3>
            {sel.bad_answer && (
              <p><b>Câu trả lời sai:</b> {sel.bad_answer}</p>
            )}
            {sel.transcript && (
              <details><summary>Transcript</summary><pre>{sel.transcript}</pre></details>
            )}
            <label>Câu trả lời đúng</label>
            <textarea
              value={sel.answer}
              onChange={(e) => setSel({ ...sel, answer: e.target.value })}
              rows={8} style={{ width: "100%" }}
            />
            <input
              value={sel.application ?? ""}
              onChange={(e) => setSel({ ...sel, application: e.target.value })}
              placeholder="application (optional)" style={{ width: "100%", marginTop: 8 }}
            />
            <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
              <button onClick={onApprove} disabled={!sel.answer.trim()}>Approve</button>
              <button onClick={onReject}>Reject</button>
            </div>
          </div>
        ) : (
          <p>Chọn một câu hỏi để xử lý.</p>
        )}
      </section>
    </main>
  );
}
