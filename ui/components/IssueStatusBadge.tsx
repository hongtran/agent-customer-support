import type { FeatureCategory, FeatureStatus, IssueStatus, RequestStatus } from "@/lib/api";
import { FEATURE_STATUSES, ISSUE_STATUSES } from "@/lib/api";

/** Which list a row belongs to. The same status word reads differently in each. */
export type RequestKind = "bug" | "feature";

export const ISSUE_STATUS_LABEL: Record<IssueStatus, string> = {
  pending: "Chờ xử lý",
  processing: "Đang xử lý",
  completed: "Đã xử lý",
  no_issue: "Không phải lỗi",
};

export const FEATURE_STATUS_LABEL: Record<FeatureStatus, string> = {
  received: "Đã tiếp nhận",
  processing: "Đang xử lý",
  responded: "Đã phản hồi",
  updated: "Đã cập nhật",
  awaiting_agreement: "Chờ thống nhất",
};

export const FEATURE_CATEGORY_LABEL: Record<FeatureCategory, string> = {
  modify: "Sửa đổi",
  upgrade: "Nâng cấp",
  extend_other_app: "Mở rộng ứng dụng khác",
};

/** The statuses an admin can pick, and their labels, per kind. */
export const STATUS_OPTIONS: Record<
  RequestKind,
  { statuses: RequestStatus[]; label: Record<string, string> }
> = {
  bug: { statuses: ISSUE_STATUSES, label: ISSUE_STATUS_LABEL },
  feature: { statuses: FEATURE_STATUSES, label: FEATURE_STATUS_LABEL },
};

const STYLE: Record<RequestStatus, string> = {
  pending: "bg-amber-50 text-amber-700 border-amber-300",
  received: "bg-amber-50 text-amber-700 border-amber-300",
  processing: "bg-blue-50 text-blue-700 border-blue-300",
  responded: "bg-violet-50 text-violet-700 border-violet-300",
  awaiting_agreement: "bg-orange-50 text-orange-700 border-orange-300",
  completed: "bg-emerald-50 text-emerald-700 border-emerald-300",
  updated: "bg-emerald-50 text-emerald-700 border-emerald-300",
  no_issue: "bg-gray-100 text-gray-600 border-gray-300",
};

export default function IssueStatusBadge({
  status,
  kind = "bug",
}: {
  status: RequestStatus;
  kind?: RequestKind;
}) {
  return (
    <span
      className={`whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium ${STYLE[status]}`}
    >
      {STATUS_OPTIONS[kind].label[status] ?? status}
    </span>
  );
}
