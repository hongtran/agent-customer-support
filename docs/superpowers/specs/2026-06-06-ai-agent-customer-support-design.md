# Thiết kế: AI Agent Hỗ trợ Khách hàng (Customer Support Agent)

- **Ngày:** 2026-06-06
- **Trạng thái:** Đã duyệt thiết kế (chờ viết implementation plan)
- **Repo:** `agent-customer-support`
- **Phụ thuộc:** `enterprise-llm-service` (RAG + LLM multi-provider)

---

## 1. Bối cảnh & Vấn đề

Công ty bán phần mềm quản lý phòng thí nghiệm (LIMS) cho các công ty tư nhân và nhà nước tại Việt Nam. Phần mềm được triển khai **on-prem trên data-center của khách hàng** vì lý do bảo mật; sau khi cài đặt, nhà cung cấp **không có quyền truy cập** hệ thống nếu không được cấp phép.

**Vấn đề:** team CS tốn nhiều công sức hướng dẫn khách đi đúng flow trong app và cách cấu hình, hiện phải dùng TeamViewer vì không kết nối được từ phía nhà cung cấp.

**Mục tiêu:** xây một AI Agent (agentic) trả lời mọi câu hỏi nghiệp vụ (bao gồm flow logic) và dẫn khách đi đúng quy trình trên web-app, nhằm giảm tải cho CS.

---

## 2. Quyết định nền tảng

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Mô hình triển khai | **Cloud agent, advisory-only** | Agent chạy trên cloud nhà cung cấp, KHÔNG kết nối hệ thống on-prem. Không có ranh giới bảo mật phải lo, dùng LLM cloud tự do. Agent *hướng dẫn* chứ không *thao tác hộ*. |
| Nền tảng tri thức | Tái dùng `enterprise-llm-service` | Đã có ingestion pipeline + RAG (Qdrant) + LLM multi-provider. Không dựng lại. |
| Framework agent | **Tự viết agent loop** sau interface mỏng | Agent bounded (4 tool); codebase hiện không dùng LangChain; gọn, dễ debug, nhất quán. |
| LLM provider | **Tái dùng layer multi-provider** của `enterprise-llm-service` | Bắt đầu tool-use với Anthropic + OpenAI; mở rộng sau. Provider đặt sau interface để đổi. |
| Biểu diễn flow | **Agentic hybrid** (flow cấu trúc + RAG fallback) | Flow cấu trúc cho ~5-10 quy trình đau nhất; RAG fallback cho phần còn lại. Tin cậy ở chỗ quan trọng, ra mắt nhanh. |
| Channel MVP | **Web widget trước**, Zalo sau | Core channel-agnostic; widget mang sẵn `customer_id` nên đơn giản. |
| Authoring flow | **Track riêng** (ngoài Spec 1) | Spec 1 chỉ cần Flow Store + Import API; admin UI / doc→flow / video→flow làm sau. |

---

## 3. Mô hình Tenancy (quan trọng)

- **Một** RAG collection duy nhất = tri thức về **sản phẩm** (không phải data riêng từng khách).
- Sản phẩm có **nhiều module**; mỗi khách (tenant) bật **một tập module** tùy nhu cầu.
- Khác nhau giữa các khách: **module bật**, **config**, **flow** (có thể override riêng).

**Hệ quả:**

- Cần **Customer Registry** mới: `customer_id → { name, enabled_modules[], config_notes, zalo_link? }`.
- `search_knowledge` query collection chung nhưng **scope theo `enabled_modules`** của khách → agent không bao giờ chỉ dẫn module khách không có.
- **Module-scoping 2 tầng** (vì knowledge hiện CHƯA gắn module metadata):
  - *Soft-scoping (chạy ngay):* đưa `enabled_modules` vào system prompt làm chỉ dẫn cho agent.
  - *Hard-filter (bật sau):* lọc metadata khi truy vấn, sau khi ingestion gắn `module`. Agent degrade mượt nếu metadata chưa sẵn.

---

## 4. Kiến trúc tổng thể

```
   Web Widget ──▶ Channel Adapters ──▶ Agent Core (loop + tools) ──┬─ search_knowledge ──▶ enterprise-llm-service
   (Zalo: sau)      (normalize I/O)         │                       │                        POST /rag/query (Qdrant)
                                            │                       ├─ list_flows / get_flow ─▶ Flow Store (DynamoDB)
                                            │                       └─ escalate_to_human ─────▶ Nhóm Zalo CS
                                            │
                                            ├─ Session Store (Redis: flow state, bước hiện tại)
                                            ├─ Conversation Store (DynamoDB: lịch sử, citation, feedback)
                                            └─ Customer Registry (DynamoDB: enabled_modules, config)
```

### Thành phần

| Thành phần | Nhiệm vụ | Phụ thuộc |
|---|---|---|
| **Channel Adapters** | Chuẩn hoá message vào/ra theo kênh; map user → `customer_id`. MVP: widget (REST/WebSocket). | FastAPI |
| **Agent Core** | Agent loop LLM + tool-use; giữ ngữ cảnh; quyết định trả lời thẳng / RAG / dẫn flow. | LLM tool client, Tool Registry |
| **Tool Registry** | 4 tool, mỗi tool một interface gọn (mô tả §5). | RAG HTTP, Flow Store, Customer Registry |
| **Flow Store** | Lưu & cung cấp playbook JSON; Import + CRUD API. | DynamoDB |
| **Session Store** | Trạng thái phiên ngắn hạn (flow đang chạy, bước hiện tại), có TTL. | Redis |
| **Conversation Store** | Lịch sử hội thoại + citation + feedback + escalation. | DynamoDB |
| **Customer Registry** | `customer_id → enabled_modules/config`. | DynamoDB |

### Hai loại lời gọi LLM (tách bạch)

1. **Gọi có tool (agent loop)** → hàm mới `ai_completion_with_tools(...)` **bổ sung vào `enterprise-llm-service`** (`llm_inference`): nhận `tools` (JSON schema), trả về `tool_use` có cấu trúc, chuẩn hoá khác biệt Anthropic ↔ OpenAI. *(Task tiền đề — `ai_completion` hiện tại chỉ trả text, không hỗ trợ tool-use.)*
2. **Gọi thuần text** (reformulate query, synthesize, tóm tắt) → **tái dùng `ai_completion`** nguyên trạng.

> Lý do bổ sung vào `enterprise-llm-service` thay vì viết LLM client riêng: giữ toàn bộ logic provider ở một chỗ, đúng tinh thần tái dùng layer multi-provider. Agent repo import wheel `enterprise_llm_service`.

---

## 5. Agent Core & Tools

### Vòng lặp xử lý 1 turn

```
AgentCore.handle_turn(session, user_msg):
  1. Nạp lịch sử hội thoại (Conversation Store) + flow state (Session Store) + hồ sơ khách (Customer Registry)
  2. Dựng messages + system prompt (gồm enabled_modules + flow state) + prompt caching + TOOLS
  3. Vòng lặp tool-use (ai_completion_with_tools):
       - tool_use → Tool Registry dispatch → tool_result → lặp
       - text     → câu trả lời cuối
  4. Lưu turn + citations + flow state mới
  5. Trả message chuẩn hoá về Channel Adapter
```

### Bốn tool

| Tool | Input | Output | Ghi chú |
|---|---|---|---|
| `search_knowledge` | `query`, `customer_id` | đoạn doc + citation | Gọi `/rag/query` HTTP; scope theo `enabled_modules` (soft → hard). |
| `list_flows` | `customer_id` | danh sách flow (id + mô tả) | Lọc theo module khách bật + override riêng. |
| `get_flow` | `flow_id` | playbook có cấu trúc | Nạp vào ngữ cảnh để dẫn từng bước. |
| `escalate_to_human` | `reason`, `transcript` | handoff record | Ghi DynamoDB + thông báo nhóm Zalo CS kèm transcript + tóm tắt. |

---

## 6. Biểu diễn Flow (Playbook)

Flow lưu trong **Flow Store (DynamoDB)** dạng JSON; seed MVP qua **Import API (JSON/YAML)**. Bắt đầu ~5-10 quy trình đau nhất, mở rộng dần.

### Schema (ý niệm)

```yaml
id: tao-mau-xet-nghiem
title: "Tạo mẫu xét nghiệm mới"
description: "Hướng dẫn tạo và gán mẫu xét nghiệm cho bệnh nhân"   # dùng cho list_flows + intent match
module: xet-nghiem                 # gắn module → lọc theo enabled_modules
scope: global                      # 'global' (mặc định sản phẩm) | '<customer_id>' (override riêng)
version: 1
language: vi
triggers: ["tạo mẫu xét nghiệm", "thêm mẫu mới"]
steps:
  - id: chon-khoa
    say: "Vào **Xét nghiệm → Quản lý mẫu**, nhấn **Tạo mới**. Bạn thấy màn hình đó chưa?"
    next:
      - when: "user xác nhận thấy màn hình"
        goto: nhap-thong-tin
      - when: "user không tìm thấy menu"
        goto: kiem-tra-quyen
  - id: kiem-tra-quyen
    say: "Có thể tài khoản chưa có quyền. Nhờ admin cấp vai trò **Kỹ thuật viên XN** ở **Cài đặt → Phân quyền**."
    next:
      - when: "đã cấp quyền"
        goto: chon-khoa
      - when: "vẫn lỗi"
        goto: escalate
  - id: nhap-thong-tin
    say: "Chọn loại mẫu, nhập mã bệnh nhân, rồi **Lưu**."
    next:
      - when: "lưu thành công"
        goto: done
      - when: "báo lỗi trùng mã"
        goto: xu-ly-trung-ma
outcomes:
  done:     { type: success,  say: "Xong! Mẫu đã được tạo." }
  escalate: { type: escalate, reason: "Không cấp được quyền tạo mẫu" }
```

> **Lưu ý:** cơ chế `refs` (liên kết bước flow → doc RAG để dẫn chứng) **bị bỏ khỏi MVP** cho gọn.

### Cách agent dùng flow (flow = guardrail, LLM lái)

1. Câu hỏi khớp `triggers` → agent gọi `get_flow`, nạp playbook.
2. Trình bày **một bước mỗi lần** (`say`).
3. **Bước hiện tại lưu ở Session Store (Redis)**; mỗi turn re-inject "đang ở bước X của flow Y" → không trôi, không nhảy bước.
4. User trả lời → LLM diễn giải khớp `when` nào → `goto` bước kế (rẽ nhánh linh hoạt theo ngôn ngữ tự nhiên).
5. Tới `outcome`: `success` → kết thúc; `escalate` → gọi `escalate_to_human`.
6. User hỏi lạc đề giữa flow → agent trả lời (RAG) rồi quay lại đúng bước.

**Lý do "LLM lái trong khung" thay vì engine tất định tuyệt đối:** schema đảm bảo thứ tự + các nhánh hợp lệ; LLM lo phần hiểu câu trả lời tự nhiên (đa dạng tiếng Việt) — đúng tinh thần hybrid.

---

## 7. Channels, Session, Escalation

### Channel Adapters

| Kênh | Cơ chế | Tenant resolution | Phạm vi |
|---|---|---|---|
| **Web widget** | JS snippet nhúng web-app on-prem, cấu hình `customer_key` + endpoint; REST (hoặc WebSocket để stream). | Widget mang sẵn `customer_id`. | **Spec 1** |
| **Zalo OA** | Webhook (verify chữ ký) + Send API. | Cần cơ chế nhận diện user → customer (mã liên kết / OA riêng). | Sau Spec 1 |

### Session & Conversation

- **Session Store (Redis):** `session_id → { flow đang chạy, bước hiện tại, last_activity }`, TTL.
- **Conversation Store (DynamoDB):** lịch sử đầy đủ + citation + feedback + escalation events.

### Escalation → human handoff

- **Kích hoạt khi:** retrieval confidence thấp, lặp lại thất bại, user xin gặp người, flow tới outcome `escalate`, dấu hiệu bực bội.
- **Hành động (MVP):** ghi handoff record (DynamoDB) + **thông báo nhóm Zalo CS** kèm transcript + tóm tắt; phía user hiện "đang chuyển cho nhân viên".
- *Live takeover (CS chen vào real-time): để sau.*

---

## 8. Tech Stack & Deployment

- **Ngôn ngữ/Framework:** Python 3.13 + FastAPI + Poetry (đồng bộ `enterprise-llm-service`), Docker.
- **Phụ thuộc:** import wheel `enterprise_llm_service` (cho `ai_completion` / `ai_completion_with_tools`); gọi `/rag/query` qua HTTP.
- **Hạ tầng:** Redis (session/flow state), DynamoDB (Conversation Store + Customer Registry + Flow Store).
- **Triển khai:** cloud phía nhà cung cấp.

---

## 9. Observability & Feedback

- **Tracing/logging** (tái dùng `utils/otel.py` của `enterprise-llm-service`): mọi turn, tool call, retrieval confidence, escalation.
- **Metrics cốt lõi (ROI):** deflection rate (giải quyết không cần người), flow completion rate, latency, cost/conversation.
- **Feedback loop (MVP nhẹ):** thu thumbs up/down + escalation → review thủ công để cải thiện flow/tri thức.

---

## 10. Chiến lược Testing (đề xuất)

- **Unit:** tool dispatch, chuyển bước flow (state transitions), module-scoping filter, customer resolution.
- **Integration:** agent loop với LLM mock (chuỗi tool tất định) + RAG mock; widget REST contract.
- **Eval set (quan trọng):** bộ kịch bản Q&A + flow tiêu biểu kèm hành vi kỳ vọng (golden tests) để chống hồi quy chất lượng agent.

---

## 11. Task tiền đề (repo `enterprise-llm-service`)

1. **`ai_completion_with_tools`** — hỗ trợ tool-use, chuẩn hoá Anthropic + OpenAI (trả về `tool_use`/`tool_result`).
2. **Module-tagging khi ingest** + tham số **filter theo module** cho `/rag/query` (bật hard-scoping).

---

## 12. Phạm vi Spec 1 (chốt) & Ngoài phạm vi

**Trong Spec 1:**
- Agent Core (loop tự viết + 4 tool), system prompt + prompt caching, soft module-scoping.
- Flow Store (DynamoDB) + Import/CRUD API + flow engine (guardrail).
- Customer Registry.
- Channel: Web widget (REST/WebSocket).
- Session Store + Conversation Store.
- Escalation → nhóm Zalo CS.
- Observability cơ bản + eval set khởi điểm.
- Task tiền đề ở `enterprise-llm-service` (mục §11).

**Ngoài phạm vi (track/spec sau):**
- Authoring tools: doc→flow (LLM), admin UI cho CS, video→flow.
- Kênh Zalo + cơ chế nhận diện customer cho Zalo.
- Hard module-filter end-to-end (phụ thuộc re-ingest).
- Live takeover, feedback loop tự động.
- Vision/multimodal (screenshot/screen-stream) — Phase 2/3 roadmap.
