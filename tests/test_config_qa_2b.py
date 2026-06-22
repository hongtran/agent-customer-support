from agent_customer_support.config import Settings


def test_qa_lead_threshold_default():
    assert Settings().qa_lead_threshold == 0.85
