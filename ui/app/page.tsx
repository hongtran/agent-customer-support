"use client";

import { useState } from "react";
import ConfigBar from "@/components/ConfigBar";
import MessageList, { Message } from "@/components/MessageList";
import InputBar from "@/components/InputBar";
import { sendMessage, sendFeedback, Attachment } from "@/lib/api";

export default function Home() {
  const [customerId, setCustomerId] = useState("ttp");
  const [conversationId, setConversationId] = useState("smoke-ui");
  const [selectedApplications, setSelectedApplications] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const handleFeedbackDown = (messageId: string) => {
    sendFeedback(conversationId, messageId);
  };

  const handleNewConversation = () => {
    setConversationId(crypto.randomUUID());
    setMessages([]);
    setSelectedApplications([]);
  };

  const handleSend = async (text: string, attachments: Attachment[]) => {
    const userContent = text + (attachments.length > 0 ? ` 📎 ${attachments.length} image${attachments.length > 1 ? "s" : ""}` : "");
    setMessages((prev) => [...prev, { role: "user", content: userContent }]);
    setLoading(true);
    try {
      const result = await sendMessage({
        customer_id: customerId,
        conversation_id: conversationId,
        message: text,
        attachments: attachments.length > 0 ? attachments : undefined,
        applications: selectedApplications.length > 0 ? selectedApplications : undefined,
      });
      setMessages((prev) => [...prev, { role: "agent", content: result.reply, messageId: result.message_id }]);
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
        selectedApplications={selectedApplications}
        onCustomerIdChange={setCustomerId}
        onConversationIdChange={setConversationId}
        onApplicationsChange={setSelectedApplications}
        onNewConversation={handleNewConversation}
      />
      <MessageList messages={messages} loading={loading} onFeedbackDown={handleFeedbackDown} />
      <InputBar onSend={handleSend} disabled={loading} />
    </div>
  );
}
