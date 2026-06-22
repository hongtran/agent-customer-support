from agent_customer_support.agents.prompts import (
    KNOWLEDGE_COMPOSE_PROMPT,
    KNOWLEDGE_COMPOSE_PROMPT_WITH_QA,
)


def test_with_qa_prompt_is_three_source_variant():
    p = KNOWLEDGE_COMPOSE_PROMPT_WITH_QA
    # third source introduced
    assert "ba nguồn" in p
    assert "ĐÁP ÁN CS XÁC NHẬN" in p
    # three-tier precedence tokens present
    assert "ưu tiên cao nhất" in p
    assert "bổ trợ" in p
    # miss marker updated to all sources
    assert "Tất cả các nguồn" in p
    # anti-hallucination updated
    assert "ngoài ba nguồn" in p
    # two-source phrasing must NOT remain in the variant
    assert "hai nguồn" not in p


def test_base_prompt_unchanged_is_two_source():
    # the original prompt stays two-source (no regression to the default path)
    assert "hai nguồn" in KNOWLEDGE_COMPOSE_PROMPT
    assert "ĐÁP ÁN CS XÁC NHẬN" not in KNOWLEDGE_COMPOSE_PROMPT
