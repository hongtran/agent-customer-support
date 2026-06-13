from dataclasses import dataclass
from typing import Any

from agent_customer_support.config import get_settings
from agent_customer_support.models import CustomerProfile
from agent_customer_support.observability import tracing

TOOL_DEFS: list[dict] = [
    {
        "name": "search_knowledge",
        "description": "Tìm trong tài liệu sản phẩm CenLab để trả lời câu hỏi nghiệp vụ.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "list_flows",
        "description": "Liệt kê các quy trình (flow) khả dụng cho khách hàng hiện tại.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_flow",
        "description": "Lấy chi tiết một quy trình theo flow_id để dẫn người dùng từng bước.",
        "input_schema": {
            "type": "object",
            "properties": {"flow_id": {"type": "string"}},
            "required": ["flow_id"],
        },
    },
    {
        "name": "log_request",
        "description": (
            "Ghi nhận yêu cầu KHÔNG trả lời được từ tài liệu. Dùng khi: "
            "(1) câu hỏi không có đáp án trong tài liệu → type='how_to_missing'; "
            "(2) khách yêu cầu thêm tính năng/đổi quy tắc → type='feature'; "
            "(3) khách báo lỗi phần mềm → type='bug'. "
            "BẮT BUỘC gọi tool này thay vì bịa câu trả lời khi không có thông tin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["feature", "bug", "how_to_missing"]},
                "summary": {"type": "string"},
                "module": {"type": "string"},
            },
            "required": ["type", "summary"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Chuyển hội thoại cho nhân viên CS khi không tự xử lý được hoặc người dùng yêu cầu.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]


@dataclass
class ToolContext:
    customer: CustomerProfile
    rag: Any
    flow_store: Any
    backlog: Any
    escalator: Any
    conversation_id: str
    transcript: str = ""
    last_fetched_flow: Any = None   # set by get_flow → enables flow activation in core


async def _dispatch(name: str, args: dict, ctx: ToolContext) -> dict:
    if name == "search_knowledge":
        return await ctx.rag.search(args["query"], collection=get_settings().product_collection)

    if name == "list_flows":
        flows = await ctx.flow_store.list_for_modules(ctx.customer.enabled_modules)
        return {"flows": [{"id": f.id, "title": f.title, "description": f.title} for f in flows]}

    if name == "get_flow":
        flow = await ctx.flow_store.get(args["flow_id"])
        if not flow:
            return {"error": "flow_not_found"}
        # Track so core.py can activate this flow when agent emits [[goto:step_id]]
        ctx.last_fetched_flow = flow
        return {"flow": flow.model_dump(mode="json")}

    if name == "log_request":
        rec = await ctx.backlog.add(
            customer_id=ctx.customer.customer_id,
            type=args["type"],
            summary=args["summary"],
            module=args.get("module"),
            transcript=ctx.transcript,
        )
        return {"logged": True, "request_id": rec.id}

    if name == "escalate_to_human":
        await ctx.escalator.escalate(
            customer_id=ctx.customer.customer_id,
            reason=args["reason"],
            transcript=ctx.transcript,
        )
        return {"escalated": True}

    return {"error": f"unknown_tool:{name}"}


async def dispatch(name: str, args: dict, ctx: ToolContext) -> dict:
    with tracing.span(f"tool.{name}", input=args) as sp:
        result = await _dispatch(name, args, ctx)
        sp.update(output=result)
        return result
