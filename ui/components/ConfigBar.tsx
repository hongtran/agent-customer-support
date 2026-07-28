// ui/components/ConfigBar.tsx
"use client";

import { useEffect, useState } from "react";
import { getCustomerApplications } from "@/lib/api";

interface Props {
  customerId: string;
  conversationId: string;
  selectedApplications: string[];
  onCustomerIdChange: (v: string) => void;
  onConversationIdChange: (v: string) => void;
  onApplicationsChange: (applications: string[]) => void;
  onNewConversation: () => void;
}

export default function ConfigBar({
  customerId,
  conversationId,
  selectedApplications,
  onCustomerIdChange,
  onConversationIdChange,
  onApplicationsChange,
  onNewConversation,
}: Props) {
  const [availableApplications, setAvailableApplications] = useState<string[]>([]);

  useEffect(() => {
    if (!customerId) return;
    getCustomerApplications(customerId).then((apps) => {
      setAvailableApplications(apps);
      // Clear any previously selected applications that no longer exist for this customer
      onApplicationsChange(selectedApplications.filter((a) => apps.includes(a)));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [customerId]);

  const toggleApplication = (app: string) => {
    if (selectedApplications.includes(app)) {
      onApplicationsChange(selectedApplications.filter((a) => a !== app));
    } else {
      onApplicationsChange([...selectedApplications, app]);
    }
  };

  return (
    <div className="border-b border-gray-200 bg-gray-50 px-4 py-2 text-sm">
      <div className="flex items-center gap-3">
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

      {availableApplications.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span className="text-xs text-gray-500">Applications:</span>
          {availableApplications.map((app) => {
            const checked = selectedApplications.includes(app);
            return (
              <button
                key={app}
                onClick={() => toggleApplication(app)}
                className={`rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors ${
                  checked
                    ? "border-blue-500 bg-blue-100 text-blue-700"
                    : "border-gray-300 bg-white text-gray-500 hover:border-gray-400"
                }`}
              >
                {checked && <span className="mr-1">✓</span>}
                {app}
              </button>
            );
          })}
          {selectedApplications.length > 0 && (
            <button
              onClick={() => onApplicationsChange([])}
              className="text-xs text-gray-400 hover:text-gray-600"
            >
              Clear
            </button>
          )}
        </div>
      )}
    </div>
  );
}
