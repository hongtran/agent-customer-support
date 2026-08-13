// ui/components/ConfigBar.tsx
"use client";

import { useEffect } from "react";
import Link from "next/link";
import { Me } from "@/lib/api";

interface Props {
  me: Me;
  conversationId: string;
  selectedApplications: string[];
  onConversationIdChange: (v: string) => void;
  onApplicationsChange: (applications: string[]) => void;
  onNewConversation: () => void;
  onLogout: () => void;
}

/**
 * Whether the user still owes us an application choice before they can ask anything.
 *
 * Only a genuine choice is gated. With exactly one application there is nothing to
 * decide — it is selected automatically below — and with none there is nothing to
 * pick at all, so gating either case would lock a customer out of their own agent.
 * `CustomerProfile.enabled_applications` is allowed to be empty, so that second case
 * is real and not defensive.
 */
export function needsApplicationChoice(available: string[], selected: string[]): boolean {
  return available.length > 1 && selected.length === 0;
}

export default function ConfigBar({
  me,
  conversationId,
  selectedApplications,
  onConversationIdChange,
  onApplicationsChange,
  onNewConversation,
  onLogout,
}: Props) {
  // The customer used to be a free-text input; it now comes from the token, so the
  // available applications come straight off the session with no extra fetch.
  const availableApplications = me.enabled_applications;
  const mustChoose = needsApplicationChoice(availableApplications, selectedApplications);

  useEffect(() => {
    // Drop any selection that isn't offered to this customer.
    const pruned = selectedApplications.filter((a) => availableApplications.includes(a));
    // A single application is a foregone conclusion, so make it for the user rather
    // than blocking the composer on a one-item decision.
    if (availableApplications.length === 1 && pruned.length === 0) {
      onApplicationsChange([availableApplications[0]]);
      return;
    }
    if (pruned.length !== selectedApplications.length) onApplicationsChange(pruned);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableApplications]);

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
        <span className="text-gray-600">
          {me.name}
          <span className="ml-1 font-mono text-xs text-gray-400">({me.customer_id})</span>
        </span>
        <label className="flex items-center gap-1 text-gray-600">
          Conv
          <input
            className="ml-1 w-48 rounded border border-gray-300 px-2 py-1 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-blue-400"
            value={conversationId}
            onChange={(e) => onConversationIdChange(e.target.value)}
          />
        </label>
        <div className="ml-auto flex items-center gap-2">
          {me.role === "admin" && (
            <Link
              href="/admin"
              className="rounded bg-gray-200 px-3 py-1 text-xs text-gray-700 hover:bg-gray-300"
            >
              Admin
            </Link>
          )}
          <button
            onClick={onNewConversation}
            className="rounded bg-gray-200 px-3 py-1 text-xs text-gray-700 hover:bg-gray-300"
          >
            Hộp thoại mới
          </button>
          <button
            onClick={onLogout}
            className="rounded px-3 py-1 text-xs text-gray-500 hover:text-gray-700"
          >
            Đăng xuất
          </button>
        </div>
      </div>

      {availableApplications.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-gray-600">
            Chọn ứng dụng
            {availableApplications.length > 1 && <span className="ml-0.5 text-rose-500">*</span>}
          </span>
          {availableApplications.map((app) => {
            const checked = selectedApplications.includes(app);
            return (
              <button
                key={app}
                onClick={() => toggleApplication(app)}
                aria-pressed={checked}
                className={`rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors ${
                  checked
                    ? "border-blue-500 bg-blue-100 text-blue-700"
                    : mustChoose
                      ? // Nothing is chosen yet and the composer is locked because of it,
                        // so the chips carry the emphasis until one is picked.
                        "border-amber-400 bg-amber-50 text-amber-800 hover:border-amber-500"
                      : "border-gray-300 bg-white text-gray-500 hover:border-gray-400"
                }`}
              >
                {checked && <span className="mr-1">✓</span>}
                {app}
              </button>
            );
          })}
          {availableApplications.length > 1 &&
            (selectedApplications.length === availableApplications.length ? (
              <button
                onClick={() => onApplicationsChange([])}
                className="text-xs text-gray-400 hover:text-gray-600"
              >
                Bỏ chọn tất cả
              </button>
            ) : (
              // Selecting everything is kept as an explicit option because an empty
              // `applications` list already means "search my whole scope" server-side.
              // Without it, the gate would only ever narrow retrieval versus today.
              <button
                onClick={() => onApplicationsChange(availableApplications)}
                className="text-xs text-blue-600 hover:text-blue-700"
              >
                Tất cả ứng dụng của tôi
              </button>
            ))}
          {mustChoose && (
            <span className="text-xs text-amber-700">
              Chọn ít nhất một ứng dụng để trợ lý tìm đúng tài liệu.
            </span>
          )}
        </div>
      )}
    </div>
  );
}
