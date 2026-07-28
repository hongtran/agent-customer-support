"use client";

import { useEffect, useRef, useState } from "react";
import { Attachment } from "@/lib/api";

interface Props {
  onSend: (text: string, attachments: Attachment[]) => void;
  disabled: boolean;
}

async function fileToBase64(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

export default function InputBar({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");
  const [pending, setPending] = useState<Attachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, [value]);

  const submit = () => {
    const text = value.trim();
    if ((!text && pending.length === 0) || disabled) return;
    onSend(text, pending);
    setValue("");
    setPending([]);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length === 0) return;
    const converted = await Promise.all(
      files.map(async (file) => ({
        kind: "image" as const,
        media_type: file.type,
        data: await fileToBase64(file),
      }))
    );
    setPending((prev) => [...prev, ...converted]);
    e.target.value = "";
  };

  const removeAttachment = (index: number) => {
    setPending((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="border-t border-gray-200 bg-white px-4 py-3">
      {pending.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {pending.map((att, i) => (
            <span
              key={i}
              className="flex items-center gap-1 rounded-lg border border-gray-300 bg-gray-50 px-2 py-1 text-xs text-gray-700"
            >
              📎 {att.media_type.replace("image/", "")}
              <button
                onClick={() => removeAttachment(i)}
                className="ml-1 text-gray-400 hover:text-gray-700"
                aria-label="Remove attachment"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={handleFileChange}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          title="Attach image"
          className="rounded-xl border border-gray-300 px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 disabled:opacity-50"
        >
          📎
        </button>
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={disabled}
          placeholder="Type a message…"
          className="flex-1 resize-none overflow-y-auto rounded-xl border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50"
          style={{ maxHeight: "120px" }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <button
          onClick={submit}
          disabled={disabled || (value.trim() === "" && pending.length === 0)}
          className="rounded-xl bg-blue-500 px-4 py-2 text-sm text-white hover:bg-blue-600 disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}
