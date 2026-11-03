"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  FeatureDetail,
  FeatureUpload,
  createFeatureRequest,
  getMyApplications,
  getMyFeatureRequest,
  listMyFeatureRequests,
} from "@/lib/api";
import { logout, useSession } from "@/lib/useSession";
import IssueView from "@/components/IssueView";

/**
 * Allowed files: extension -> media type. Same list as the server's `_ALLOWED_FILES`
 * (channels/feature_requests.py). The media type is taken from the extension, not from
 * the browser, because browsers report Office files inconsistently (often as "").
 */
const FILE_TYPES: Record<string, string> = {
  ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ".xls": "application/vnd.ms-excel",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".doc": "application/msword",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
};
const ACCEPT = Object.keys(FILE_TYPES).join(",");
const MAX_FILES = 5;
/** Mirrors the server's MAX_ATTACHMENT_BYTES default (5 MB). The server checks again. */
const MAX_FILE_BYTES = 5 * 1024 * 1024;

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i < 0 ? "" : name.slice(i).toLowerCase();
}

/** A file's bytes as base64 (the data-URL prefix removed). */
function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] ?? "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/** Why these files cannot be sent, or "" when they can. */
function checkFiles(files: File[]): string {
  if (files.length > MAX_FILES) return `Tối đa ${MAX_FILES} file.`;
  for (const f of files) {
    if (!FILE_TYPES[extOf(f.name)])
      return `"${f.name}": chỉ nhận file Excel, Word hoặc ảnh (png, jpg).`;
    if (f.size > MAX_FILE_BYTES) return `"${f.name}" lớn hơn 5 MB.`;
  }
  return "";
}

const inputCls = "w-full rounded border border-gray-300 bg-white px-2 py-1 text-sm";

function Field({
  label,
  required,
  children,
}: {
  label: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <label className="block space-y-1 text-xs text-gray-600">
      <span>
        {label}
        {required && <span className="text-rose-600"> *</span>}
      </span>
      {children}
    </label>
  );
}

/**
 * The form for a new feature request. The customer is not a field: the server takes it
 * from the login token. The requester is: the login is shared by the whole company, so
 * CS needs to know which person asked. The application list is the customer's own.
 */
function CreateForm({
  onCreated,
  onCancel,
}: {
  onCreated: (rec: FeatureDetail) => void;
  onCancel: () => void;
}) {
  const [apps, setApps] = useState<string[]>([]);
  const [title, setTitle] = useState("");
  const [name, setName] = useState("");
  const [position, setPosition] = useState("");
  const [department, setDepartment] = useState("");
  const [application, setApplication] = useState("");
  const [summary, setSummary] = useState("");
  const [reason, setReason] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getMyApplications().then((a) => {
      setApps(a);
      // One application: nothing to choose, so pre-select it.
      if (a.length === 1) setApplication(a[0]);
    });
  }, []);

  const fileError = checkFiles(files);
  const valid =
    [title, name, position, department, summary].every((v) => v.trim() !== "") &&
    application !== "" &&
    !fileError;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!valid) return;
    setSaving(true);
    try {
      setError("");
      const uploads: FeatureUpload[] = await Promise.all(
        files.map(async (f) => ({
          name: f.name,
          media_type: FILE_TYPES[extOf(f.name)],
          data: await toBase64(f),
        })),
      );
      onCreated(
        await createFeatureRequest({
          title: title.trim(),
          requester_name: name.trim(),
          requester_position: position.trim(),
          requester_department: department.trim(),
          application,
          summary: summary.trim(),
          reason: reason.trim() || null,
          files: uploads,
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <form
      onSubmit={submit}
      className="max-h-[65vh] space-y-3 overflow-y-auto border-b border-gray-200 bg-gray-50 px-4 py-3"
    >
      {error && <p className="text-xs text-rose-700">{error}</p>}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <Field label="Tên nhân sự" required>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={100}
            className={inputCls}
          />
        </Field>
        <Field label="Chức vụ" required>
          <input
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            maxLength={100}
            className={inputCls}
          />
        </Field>
        <Field label="Phòng ban" required>
          <input
            value={department}
            onChange={(e) => setDepartment(e.target.value)}
            maxLength={100}
            className={inputCls}
          />
        </Field>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="md:col-span-2">
          <Field label="Tiêu đề" required>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              maxLength={200}
              className={inputCls}
            />
          </Field>
        </div>
        <Field label="Ứng dụng" required>
          <select
            value={application}
            onChange={(e) => setApplication(e.target.value)}
            className={inputCls}
          >
            <option value="">— Chọn ứng dụng —</option>
            {apps.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </Field>
      </div>
      <Field label="Nội dung yêu cầu" required>
        <textarea
          value={summary}
          onChange={(e) => setSummary(e.target.value)}
          maxLength={4000}
          rows={4}
          placeholder="Bạn muốn làm gì, ở màn hình nào…"
          className={inputCls}
        />
      </Field>
      <Field label="Lý do">
        <textarea
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          maxLength={2000}
          rows={2}
          placeholder="Vì sao cần tính năng này (không bắt buộc)"
          className={inputCls}
        />
      </Field>
      <Field label={`File đính kèm (Excel, Word, ảnh — tối đa ${MAX_FILES} file, mỗi file ≤ 5 MB)`}>
        <input
          type="file"
          multiple
          accept={ACCEPT}
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          className="block text-xs"
        />
      </Field>
      {fileError && <p className="text-xs text-rose-700">{fileError}</p>}
      <div className="flex justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          className="rounded px-3 py-1 text-xs text-gray-500 hover:text-gray-700"
        >
          Hủy
        </button>
        <button
          type="submit"
          disabled={!valid || saving}
          className="rounded bg-blue-600 px-4 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-40"
        >
          {saving ? "Đang gửi…" : "Gửi yêu cầu"}
        </button>
      </div>
    </form>
  );
}

/** The feature requests this customer sent, and where CS has them. */
export default function MyFeatureRequestsPage() {
  const router = useRouter();
  const session = useSession();
  const [showForm, setShowForm] = useState(false);
  // Bumped after a create so IssueView reloads its list (it reloads when `loadList`
  // changes identity), then opens the new request.
  const [version, setVersion] = useState(0);
  const [openId, setOpenId] = useState<string | null>(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const loadList = useCallback(() => listMyFeatureRequests(), [version]);

  if (session.status === "loading") {
    return <div className="flex h-screen items-center justify-center text-sm text-gray-400">…</div>;
  }

  return (
    <div className="flex h-screen flex-col bg-white text-gray-800">
      <header className="flex items-center gap-3 border-b border-gray-200 bg-gray-50 px-4 py-3">
        <Link href="/" className="text-xs text-blue-600 hover:text-blue-700">
          ← Trang chat
        </Link>
        <h1 className="text-sm font-semibold text-gray-700">Yêu cầu tính năng</h1>
        {!showForm && (
          <button
            onClick={() => setShowForm(true)}
            className="rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-700"
          >
            + Tạo yêu cầu mới
          </button>
        )}
        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs text-gray-500">{session.me.name}</span>
          <button
            onClick={() => logout(router)}
            className="rounded px-3 py-1 text-xs text-gray-500 hover:text-gray-700"
          >
            Đăng xuất
          </button>
        </div>
      </header>
      {showForm && (
        <CreateForm
          onCancel={() => setShowForm(false)}
          onCreated={(rec) => {
            setShowForm(false);
            setVersion((v) => v + 1);
            setOpenId(rec.id);
          }}
        />
      )}
      <IssueView
        kind="feature"
        loadList={loadList}
        loadDetail={getMyFeatureRequest}
        openId={openId}
        emptyText="Bạn chưa gửi yêu cầu tính năng nào."
      />
    </div>
  );
}
