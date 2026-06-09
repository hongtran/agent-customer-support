"use client";

import { useState } from "react";
import ConfigBar from "@/components/ConfigBar";
import MessageList, { Message } from "@/components/MessageList";
import InputBar from "@/components/InputBar";
import { sendMessage } from "@/lib/api";

export default function Home() {
  const [customerId, setCustomerId] = useState("ttp");
  const [conversationId, setConversationId] = useState("smoke-ui");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const handleNewConversation = () => {
    setConversationId(crypto.randomUUID());
    setMessages([]);
  };

  const handleSend = async (text: string) => {
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    try {
      const result = await sendMessage({
        customer_id: customerId,
        conversation_id: conversationId,
        message: text,
      });
      setMessages((prev) => [...prev, { role: "agent", content: result.reply }]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [...prev, { role: "error", content: `Error: ${msg}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen flex-col">
      <ConfigBar
        customerId={customerId}
        conversationId={conversationId}
        onCustomerIdChange={setCustomerId}
        onConversationIdChange={setConversationId}
        onNewConversation={handleNewConversation}
      />
      <MessageList messages={messages} loading={loading} />
      <InputBar onSend={handleSend} disabled={loading} />
    </div>
  );
}
