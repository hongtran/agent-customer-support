import type { IssueStatus } from "@/lib/api";

export const ISSUE_STATUS_LABEL: Record<IssueStatus, string> = {
  pending: "Chờ xử lý",
  processing: "Đang xử lý",
  completed: "Đã xử lý",
  no_issue: "Không phải lỗi",
};

const STYLE: Record<IssueStatus, string> = {
  pending: "bg-amber-50 text-amber-700 border-amber-300",
  processing: "bg-blue-50 text-blue-700 border-blue-300",
  completed: "bg-emerald-50 text-emerald-700 border-emerald-300",
  no_issue: "bg-gray-100 text-gray-600 border-gray-300",
};

export default function IssueStatusBadge({ status }: { status: IssueStatus }) {
  return (
    <span
      className={`whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium ${STYLE[status]}`}
    >
      {ISSUE_STATUS_LABEL[status]}
    </span>
  );
}
