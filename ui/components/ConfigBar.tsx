// ui/components/ConfigBar.tsx
"use client";

interface Props {
  customerId: string;
  conversationId: string;
  onCustomerIdChange: (v: string) => void;
  onConversationIdChange: (v: string) => void;
  onNewConversation: () => void;
}

export default function ConfigBar({
  customerId,
  conversationId,
  onCustomerIdChange,
  onConversationIdChange,
  onNewConversation,
}: Props) {
  return (
    <div className="flex items-center gap-3 border-b border-gray-200 bg-gray-50 px-4 py-2 text-sm">
      <label className="flex items-center gap-1 text-gray-600">
        Customer
        <input
          className="ml-1 rounded border border-gray-300 px-2 py-1 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
          value={customerId}
          onChange={(e) => onCustomerIdChange(e.target.value)}
        />
      </label>
      <label className="flex items-center gap-1 text-gray-600">
        Conv
        <input
          className="ml-1 w-48 rounded border border-gray-300 px-2 py-1 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
          value={conversationId}
          onChange={(e) => onConversationIdChange(e.target.value)}
        />
      </label>
      <button
        onClick={onNewConversation}
        className="ml-auto rounded bg-gray-200 px-3 py-1 text-xs text-gray-700 hover:bg-gray-300"
      >
        New conversation
      </button>
    </div>
  );
}
