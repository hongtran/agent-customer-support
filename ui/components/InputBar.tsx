"use client";

import { useRef } from "react";

interface Props {
  onSend: (text: string) => void;
  disabled: boolean;
}

export default function InputBar({ onSend, disabled }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const text = ref.current?.value.trim() ?? "";
    if (!text || disabled) return;
    onSend(text);
    if (ref.current) ref.current.value = "";
  };

  return (
    <div className="flex items-end gap-2 border-t border-gray-200 bg-white px-4 py-3">
      <textarea
        ref={ref}
        rows={1}
        disabled={disabled}
        placeholder="Type a message…"
        className="flex-1 resize-none rounded-xl border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50"
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <button
        onClick={submit}
        disabled={disabled}
        className="rounded-xl bg-blue-500 px-4 py-2 text-sm text-white hover:bg-blue-600 disabled:opacity-50"
      >
        Send
      </button>
    </div>
  );
}
