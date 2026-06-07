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
| Framework agent | **Tự viết agent loop** sau interface mỏng | Agent bounded (5 tool); codebase hiện không dùng LangChain; gọn, dễ debug, nhất quán. |
| LLM provider | **Tái dùng layer multi-provider** của `enterprise-llm-service` | Bắt đầu tool-use với Anthropic + OpenAI; mở rộng sau. Provider đặt sau interface để đổi. |
| Biểu diễn flow | **Agentic hybrid** (flow cấu trúc + RAG fallback) | Flow cấu trúc cho ~5-10 quy trình đau nhất; RAG fallback cho phần còn lại. Tin cậy ở chỗ quan trọng, ra mắt nhanh. |
| Channel MVP | **Web widget trước**, Zalo sau | Core channel-agnostic; widget mang sẵn `customer_id` nên đơn giản. |
| Authoring flow | **Track riêng** (ngoài Spec 1) | Spec 1 chỉ cần Flow Store + Import API; admin UI / doc→flow / video→flow làm sau. |

---

## 2.1. Phân tích workload CS (từ dữ liệu thật)

Phân tích file `1. Cac yeu cau TTP-Cenlab 2026.xlsx` (danh sách yêu cầu thật của 1 khách, cột **"Phân loại"** do Tâm Đức gán) cho thấy workload chia gần **50/50** thành hai loại:

| Phân loại | Ví dụ thật trong file | Agent xử lý? |
|---|---|---|
| **"Hướng dẫn sử dụng"** | "Tên mẫu nhập thế nào?", "copy đơn hàng", "phân quyền PKD đi lấy mẫu", "chọn ngày quá khứ" | ✅ RAG + flow |
| **"Nâng cấp"** (feature request) | "bổ sung cột địa điểm lấy mẫu", "thêm biên bản lấy mẫu QT", "đổi quy tắc mã mẫu" | ❌ cần dev — chỉ triage/định tuyến |

**Hệ quả thiết kế — triage theo kiểu "try-then-route" (KHÔNG phải classifier trả trước):** PoC cho thấy phân loại how-to/feature trả trước chỉ đạt ~60% so với người (§13), vì ranh giới phụ thuộc việc *biết phần mềm làm được gì* — kiến thức có được nhờ **search**, không phải đoán mù. Vì vậy agent **thử giải quyết trước** (`search_knowledge` + `get_flow`); chỉ khi *không tìm được đáp án có căn cứ* mới đề xuất `log_request` (feature/bug) + escalate. Không bịa quy trình không tồn tại.

**Kỳ vọng thực tế:** agent không giải quyết ~nửa số yêu cầu (feature request thật); giá trị = deflect nhóm "how-to" có tài liệu + tự động định tuyến phần còn lại.

**Knowledge gap (bắt buộc xử lý):** PoC chứng minh ~nửa đáp án how-to thật ("có sẵn", workaround, "tên mẫu = tên nền mẫu") **không nằm trong HDSD** mà ở ticket cũ / đầu CS. → Ingest **chỉ HDSD là chưa đủ**; phải nạp thêm **ticket đã giải quyết + chính file Excel (cột giải pháp) + tri thức tribal** vào RAG. Đây là điều kiện cần để đạt deflection rate cao.

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
- **`module` suy được từ cấu trúc tài liệu:** HDSD đánh số mục rõ (`3. PYC`, `4. NHẬN MẪU`, `5. PHÂN CÔNG`...) → enrichment có thể gán `module` theo heading khi ingest (đã kiểm chứng ở §13).

---

## 4. Kiến trúc tổng thể

```
   Web Widget ──▶ Channel Adapters ──▶ Agent Core (loop + tools) ──┬─ search_knowledge ──▶ enterprise-llm-service
   (Zalo: sau)      (normalize I/O)         │                       │                        POST /rag/query (Qdrant)
                                            │                       ├─ list_flows / get_flow ─▶ Flow Store (DynamoDB)
                                            │                       ├─ log_request ───────────▶ Request backlog (DynamoDB)
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
| **Tool Registry** | 5 tool, mỗi tool một interface gọn (mô tả §5). | RAG HTTP, Flow Store, Customer Registry |
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
  2. Dựng messages + system prompt (gồm enabled_modules + flow state + hướng dẫn TRIAGE) + prompt caching + TOOLS
  3. Vòng lặp tool-use (ai_completion_with_tools):
       - tool_use → Tool Registry dispatch → tool_result → lặp
       - text     → câu trả lời cuối
       (Triage ngầm trong loop: nếu là feature request/bug → log_request, KHÔNG bịa giải pháp)
  4. Lưu turn + citations + flow state mới
  5. Trả message chuẩn hoá về Channel Adapter
```

**Triage = "try-then-route" (§2.1):** agent KHÔNG phân loại trả trước. Nó luôn **thử `search_knowledge`/`get_flow` trước**; nếu có đáp án căn cứ → trả lời/dẫn flow. Nếu *không tìm được* (vượt khả năng phần mềm) → gọi `log_request` (feature/bug) + báo CS, **không** bịa quy trình không tồn tại.

### Năm tool

| Tool | Input | Output | Ghi chú |
|---|---|---|---|
| `search_knowledge` | `query`, `customer_id` | đoạn doc + citation | Gọi `/rag/query` HTTP; scope theo `enabled_modules` (soft → hard). |
| `list_flows` | `customer_id` | danh sách flow (id + mô tả) | Lọc theo module khách bật + override riêng. |
| `get_flow` | `flow_id` | playbook có cấu trúc | Nạp vào ngữ cảnh để dẫn từng bước. |
| `log_request` | `type` (feature/bug), `summary`, `module`, `transcript` | request record | Ghi backlog (DynamoDB) + thông báo CS/product. Phản ánh cột "Phân loại" của Tâm Đức. |
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
- **Eval set (quan trọng):** dùng **file yêu cầu thật `1. Cac yeu cau TTP-Cenlab 2026.xlsx`** làm golden eval — mỗi dòng có yêu cầu + **phân loại** (how-to vs nâng cấp) + giải pháp CS đã trả. Đo: (1) triage đúng loại, (2) deflection rate trên nhóm "Hướng dẫn sử dụng", (3) đúng flow/đáp án so với CS. Bổ sung thêm kịch bản flow tiêu biểu.

---

## 11. Task tiền đề (repo `enterprise-llm-service`)

1. **`ai_completion_with_tools`** — hỗ trợ tool-use, chuẩn hoá Anthropic + OpenAI (trả về `tool_use`/`tool_result`).
2. **Module-tagging khi ingest** + tham số **filter theo module** cho `/rag/query` (bật hard-scoping). Gán `module` theo heading đánh số của HDSD.

> **Ingest HDSD KHÔNG cần OCR mới.** `FileParser` đã dùng **Gemini multimodal** (`gemini-2.5-flash`): [`split_pdf`](../../enterprise-llm-service/enterprise_llm_service/data_processing/file_parser.py) (5 trang/chunk, overlap 1) → upload → `PROMPT_TO_CONVERT_DOC_TO_MARKDOWN`. PDF tuy là **ảnh** (không text layer) nhưng Gemini đọc được ảnh trang → markdown sạch (đã kiểm chứng §13). Pipeline còn có sẵn `PROMPT_TO_CLASSIFY_DOC_TYPE` (workflow/config_rule/faq/reference/general) và `PROMPT_TO_EXTRACT_WORKFLOW` — nền tảng cho doc→flow.

---

## 12. Phạm vi Spec 1 (chốt) & Ngoài phạm vi

**Trong Spec 1:**
- Agent Core (loop tự viết + 5 tool), system prompt + prompt caching, **triage "try-then-route"** (§2.1), soft module-scoping.
- Flow Store (DynamoDB) + Import/CRUD API + flow engine (guardrail).
- Customer Registry + Request backlog (cho `log_request`).
- Channel: Web widget (REST/WebSocket).
- Session Store + Conversation Store.
- Escalation → nhóm Zalo CS.
- **Ingest tri thức qua pipeline FILE_PARSING sẵn có** (Gemini→markdown): HDSD Cenlab **+ ticket cũ + file Excel giải pháp** (bắt buộc — §13 cho thấy ~nửa đáp án how-to không nằm trong HDSD).
- Observability cơ bản + eval set khởi điểm (từ file Excel yêu cầu thật).
- Task tiền đề ở `enterprise-llm-service` (mục §11).

**Ngoài phạm vi (track/spec sau):**
- Authoring tools: doc→flow (LLM, đã chứng minh khả thi §13), admin UI cho CS, video→flow.
- Kênh Zalo + cơ chế nhận diện customer cho Zalo.
- Hard module-filter end-to-end (phụ thuộc re-ingest).
- Live takeover, feedback loop tự động.
- Vision/multimodal (screenshot/screen-stream) — Phase 2/3 roadmap.

---

## 13. Kiểm chứng (PoC trên dữ liệu thật)

Đã chạy PoC trên `HDSD Cenlab.pdf` (76 trang, **toàn ảnh, không có text layer**) bằng chính `gemini-2.5-flash` mà pipeline đang dùng:

1. **Ingest (PDF ảnh → markdown):** trích 5 trang (19–23) → upload Gemini → `PROMPT_TO_CONVERT_DOC_TO_MARKDOWN` → markdown tiếng Việt **sạch, giữ nguyên** heading đánh số (`3.3`, `3.4 PYC sự cố`, `4. NHẬN MẪU`...), chuỗi **"Bước 1/2/3"**, đường dẫn menu và tên trường đọc từ screenshot. → xác nhận ingest không cần OCR mới.
2. **doc→flow (markdown → playbook):** đưa markdown qua prompt trích flow → ra **JSON playbook hợp lệ đúng schema §6** cho quy trình "PYC sự cố" (id/title/module/triggers/steps/next/outcomes). → xác nhận doc→flow khả thi và rẻ vì tái dùng output ingest.

### PoC sâu — ingest toàn bộ + đo trên yêu cầu thật

3. **Ingest toàn bộ 76 trang** → KB markdown ~56k ký tự (~14k token). 10 chunk Gemini, chạy gọn.
4. **Triage (so với cột "Phân loại" của Tâm Đức, 22 yêu cầu):** classifier trả trước đạt **64% zero-shot**, **59% khi có KB** — KB không cứu được. Các ca sai chủ yếu là yêu cầu *nghe như feature* nhưng thực ra tính năng "có sẵn"/có workaround mà HDSD không ghi. → đổi sang **try-then-route** (§2.1, §5).
5. **Q&A (KB làm context):** quy trình có trong HDSD (PYC sự cố, hủy PYCTN, sửa kết quả) → trả lời **đúng, có bước cụ thể**; câu không có trong HDSD ("tên mẫu nhập sao") → agent **tự nói "chưa có trong tài liệu, chuyển nhân viên"**, KHÔNG hallucination.

**Kết luận:**
- Rủi ro ingest + doc→flow + Q&A-grounded: **thấp** (đã chứng minh).
- Nút thắt thật = **độ phủ tri thức**: phải ingest ticket cũ + Excel giải pháp + tribal, không chỉ HDSD.
- Triage không nên là classifier trả trước → **try-then-route**.
- Chất lượng agent loop cuối cùng đo bằng eval set §10.
