"""Response schemas for the agents that ask the LLM for a decision, not prose.

These live in the LLM layer rather than `models.py` because they describe wire
format -- what one provider call is constrained to return -- not domain state.

A schema guarantees the *shape* of an answer, never its *correctness* and never
that a call succeeded at all: both provider `parse` APIs return an Optional. Every
caller keeps its fail-safe default for the None case.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TriageDecision(BaseModel):
    """Routing target chosen by TriageAgent.

    `flow` is deliberately absent: FlowAgent is reached from the
    `session.active_flow_id` fast path in TriageAgent.run, before any LLM call, so
    the model is never the thing that picks it.
    """

    model_config = ConfigDict(extra="forbid")

    target: Literal["knowledge", "escalate", "out_of_scope"] = Field(
        description=(
            "knowledge for any product question, how-to, bug report or feature "
            "request; escalate only when the user explicitly asks for a human; "
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
    """A composed reply plus the sources it stands on.

    Deliberately a thin envelope: `answer` carries the reply EXACTLY as the free-text
    composer used to produce it, markers and all ([[clarify]], [[no_answer]],
    [[suspected_bug:...]], [[img:...]]), and `parse_markers` still reads them out of it.
    Only the citation list is promoted to a typed field. The alternative -- modelling
    every marker as a schema field -- would mean rewriting the most heavily tuned prompt
    in the repo to buy a guarantee the markers already have.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description=(
            "Câu trả lời tiếng Việt đầy đủ cho người dùng, GIỮ NGUYÊN mọi marker "
            "([[clarify]], [[no_answer]], [[suspected_bug:...]], [[img:...]]) đúng như "
            "hướng dẫn trong system prompt."
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
    the unsupported idea. That precision is not cosmetic: `guardrail.strip_claims` deletes
    the span in Python when every claim is minor, and it can only do that if the text is
    findable exactly once. A paraphrase or a whole sentence sends the reply to the LLM
    repair path instead.
    """

    model_config = ConfigDict(extra="forbid")

    span: str = Field(
        description=(
            "Đoạn văn bản CHÉP NGUYÊN VĂN từ câu trả lời (kể cả dấu câu và khoảng trắng), "
            "ngắn nhất có thể — chỉ cụm từ chứa ý thiếu căn cứ, không phải cả câu."
        )
    )
    severity: Literal["minor", "critical"] = Field(
        description=(
            "minor: chi tiết thừa không có trong nguồn nhưng không sai và không làm người "
            "dùng thao tác khác đi (mẹo chung, ngữ cảnh vô hại). critical: bịa thông tin, không thể suy ra từ nguồn hoặc quy trình"
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
