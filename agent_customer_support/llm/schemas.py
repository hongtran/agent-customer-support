"""Response schemas for the agents that ask the LLM for a decision, not prose.

These live in the LLM layer rather than `models.py` because they describe wire
format -- what one provider call is constrained to return -- not domain state.

A schema guarantees the *shape* of an answer, never its *correctness* and never
that a call succeeded at all: both provider `parse` APIs return an Optional. Every
caller keeps its fail-safe default for the None case.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_customer_support.models import BugSlots


class TriageDecision(BaseModel):
    """Routing target chosen by TriageAgent."""

    model_config = ConfigDict(extra="forbid")

    target: Literal["knowledge", "issue_verification", "escalate", "out_of_scope"] = Field(
        description=(
            "knowledge for any product question, how-to or feature request; "
            "issue_verification ONLY when the user reports the software itself "
            "misbehaving (error message, crash, wrong result, an action that will "
            "not complete) — a how-to that merely contains the word 'lỗi' is still "
            "knowledge; escalate only when the user explicitly asks for a human; "
            "out_of_scope only when the question is clearly unrelated to the "
            "CenLab software (finance, weather, sports, food, translation, "
            "general programming...) — when in doubt, pick knowledge."
        )
    )


class CitedSource(BaseModel):
    """One source the answer used, and which part of it.

    `id` addresses the source; `section` narrows it to the heading inside that passage
    whose content actually reached the answer. A chunk often carries several headings --
    a document title followed by a subsection, say -- so which one applies is not
    decidable from the chunk alone. Only the answer knows.

    The same `id` may appear twice with different sections when the answer drew on two
    parts of one chunk. It must never appear twice with the SAME section, and `section`
    must never be a heading the answer did not use: this is a record of what was used,
    not an index of what was available.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description=(
            'Nguồn: chỉ số đoạn trích ("0", "1"...), "qa:<i>" cho đáp án CS, '
            '"quy_trinh_chung" cho QUY TRÌNH.'
        )
    )
    # Required with "" as the no-value case, not optional: OpenAI strict mode drops a
    # defaulted field from `required`.
    section: str = Field(
        description=(
            "Tiêu đề mục (heading) trong đoạn trích mà bạn thực sự lấy nội dung, chép "
            "NGUYÊN VĂN, bỏ dấu # và bỏ số thứ tự đầu. Rỗng nếu đoạn trích không có "
            'heading, hoặc id là "quy_trinh_chung".'
        )
    )


class ComposedAnswer(BaseModel):
    """A composed reply, what it decided, and the sources it stands on.

    `status` is the composer's routing decision, and it used to be a marker written
    inline in the prose ([[clarify]], [[no_answer]], [[suspected_bug:<app>]]) that
    `parse_markers` read back out with regexes. Promoting it to a field is what lets
    `routing.next_step` read a validated value instead of re-parsing text, and it is
    the last place in the pipeline where an agent signalled a decision as a string.

    The [[img:...]] markers stay inline in `answer`, because they are positional --
    they mark WHERE in the prose a screenshot belongs -- so there is nothing to
    promote. `parse_markers` survives for the free-text fallback in `_compose`, which
    has no schema to fill.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description=(
            "Câu trả lời tiếng Việt đầy đủ cho người dùng. GIỮ NGUYÊN các marker ảnh "
            "([[img:...]]) đúng vị trí. KHÔNG viết [[clarify]], [[no_answer]] hay "
            "[[suspected_bug:...]] vào đây — dùng trường status."
        )
    )
    status: Literal["answer", "clarify", "no_answer", "suspected_bug"] = Field(
        description=(
            '"answer": trả lời trực tiếp được. '
            '"clarify": cần hỏi lại/xác nhận trước khi trả lời — answer chứa câu hỏi đó. '
            '"no_answer": KHÔNG nguồn nào trả lời được — answer để rỗng. '
            '"suspected_bug": tài liệu xác nhận tính năng đáng lẽ chạy nhưng người dùng '
            "báo lỗi."
        )
    )
    # Required with "" as the no-value case, same reason as CitedSource.section.
    application: str = Field(
        description=(
            'Chỉ khi status="suspected_bug": mã phân hệ (application) gặp lỗi, dạng slug '
            "(ví dụ lay_mau_quan_trac). Rỗng trong mọi trường hợp khác."
        )
    )
    cited: list[CitedSource] = Field(
        description=(
            "Các nguồn THỰC SỰ được dùng để viết câu trả lời, kèm tiêu đề mục đã dùng. "
            "Rỗng nếu không dùng nguồn nào."
        )
    )


class UnsupportedClaim(BaseModel):
    """One claim in a reply that no cited source supports.

    `span` is a VERBATIM substring of the reply, kept to the smallest phrase that carries
    the unsupported idea. That precision is not cosmetic: `guardrail.apply_claims` edits
    the span in Python when every claim is minor, and it can only do that if the text is
    findable exactly once. A paraphrase sends the reply to the LLM repair path instead.

    `replacement` is what the span becomes -- empty to delete it outright, otherwise the
    span with the unsupported words taken out, so the sentence stays grammatical. It may
    only REUSE the span's own words in their original order (punctuation and case may
    change); `apply_claims` enforces that with a subsequence check, because a judge that
    could write free text here would be a second composer with no grounding check.
    """

    model_config = ConfigDict(extra="forbid")

    span: str = Field(
        description=(
            "Đoạn văn bản CHÉP NGUYÊN VĂN từ câu trả lời (kể cả dấu câu và khoảng trắng), "
            "ngắn nhất có thể — chỉ cụm từ chứa ý thiếu căn cứ, không phải cả câu."
        )
    )
    replacement: str = Field(
        description=(
            "Phần THAY THẾ cho span để câu còn lại đúng ngữ pháp: chỉ được dùng lại các từ "
            "có trong span theo đúng thứ tự (bỏ bớt từ, sửa dấu câu/viết hoa), TUYỆT ĐỐI "
            "không thêm từ mới. Để rỗng nếu xóa hẳn span."
        )
    )
    severity: Literal["minor"] = Field(
        description=(
            "minor: chi tiết thừa không có trong nguồn nhưng không sai và không làm người "
            "dùng thao tác khác đi (mẹo chung, ngữ cảnh vô hại)."
        )
    )
    reason: str = Field(description="lý do ngắn, tiếng Việt")


class GroundingVerdict(BaseModel):
    """Whether every claim in a reply is supported by the sources it cited.

    Narrower than the moderation verdict it replaced: this judge sees the cited
    passages and rules on grounding alone. Tone, scope and prompt-leakage are not its
    job -- scope is triage's gate, and mixing the three into one verdict is what made
    the previous single guardrail prompt hard to tune.

    There is no top-level reason: each unsupported claim carries its own, and its
    severity is what decides between a Python delete, an LLM repair and a handoff.
    """

    model_config = ConfigDict(extra="forbid")

    grounded: bool = Field(description="true when every claim is supported by the sources")
    # Required, not optional: OpenAI strict mode demands every property appear in
    # `required`, and a field with a default would be dropped from it.
    unsupported_claims: list[UnsupportedClaim] = Field(
        description="the specific claims with no support; empty list when grounded is true"
    )


class BugReport(BaseModel):
    """A verified bug, written up for the engineering tracker.

    Produced once, when IssueVerificationAgent decides the evidence is complete, and
    consumed by MantisClient. Every field is required with "" as the no-value case
    (same reason as CitedSource.section: OpenAI strict mode drops a defaulted field
    from `required`). The caller keeps a code-derived fallback for the None case.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        description=(
            "MỘT dòng tiêu đề ngắn (dưới 100 ký tự), tiếng Việt, nêu chức năng và triệu "
            "chứng lỗi. Ví dụ: 'Import ký hiệu mẫu báo lỗi 500'."
        )
    )
    summary: str = Field(
        description=(
            "2-4 câu tóm tắt cho đội kỹ thuật: người dùng làm gì, hệ thống phản hồi ra sao, "
            "người dùng mong đợi gì. CHỈ dùng thông tin có trong hội thoại."
        )
    )
    steps_to_reproduce: str = Field(
        description=(
            "Các bước tái hiện, mỗi bước một dòng đánh số, lấy từ hội thoại. Để rỗng nếu "
            "người dùng không mô tả bước nào — KHÔNG tự bịa."
        )
    )


class VerificationDecision(BaseModel):
    """One turn of bug verification: what we now know, and what to do about it.

    Replaces a free-text reply with an [[evidence_ready]] marker regexed out of it.
    The bar used to be "at least ONE of an error message, a screenshot or repro
    steps", which is why `slots` exists: the agent now reports the whole fixed set of
    facts a ticket needs, so Python can see what is still missing and the caller can
    file a usable ticket even when the collection is cut short.

    `outcome` carries the decision the old boolean could not express. "user_error"
    costs no ticket and no handoff -- the feature behaved correctly and the user needs
    an explanation, which is KnowledgeAgent's job.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: Literal["need_more_info", "user_error", "bug_confirmed"] = Field(
        description=(
            '"need_more_info": còn thiếu thông tin quan trọng — hỏi tiếp. '
            '"user_error": phần mềm chạy ĐÚNG như tài liệu, người dùng thao tác nhầm. '
            '"bug_confirmed": đã đủ thông tin và đây thực sự là lỗi phần mềm.'
        )
    )
    reply: str = Field(
        description=(
            "Tin nhắn tiếng Việt gửi người dùng: câu hỏi cho TỐI ĐA hai thông tin còn "
            "thiếu khi need_more_info; giải thích ngắn cách dùng đúng khi user_error; "
            "câu xác nhận đã ghi nhận khi bug_confirmed."
        )
    )
    slots: BugSlots = Field(
        description=(
            "Toàn bộ thông tin đã thu thập được cho đến lúc này, lấy từ CẢ hội thoại. "
            "Để rỗng từng trường chưa biết — KHÔNG bịa."
        )
    )
