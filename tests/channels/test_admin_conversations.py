from agent_customer_support.channels.admin_conversations import _applications_used
from agent_customer_support.models import Conversation, Turn


def test_applications_used_is_the_ordered_union_of_user_turns():
    turns = [
        Turn(role="user", content="a", applications=["B", "A"]),
        Turn(role="assistant", content="r"),
        Turn(role="user", content="b", applications=["A", "C"]),
    ]
    assert _applications_used(turns) == ["B", "A", "C"]


def test_turn_stored_before_the_field_loads_with_no_applications():
    conv = Conversation.model_validate(
        {
            "conversation_id": "cv1",
            "customer_id": "c1",
            "turns": [{"role": "user", "content": "old"}],
        }
    )
    assert conv.turns[0].applications == []
    assert _applications_used(conv.turns) == []
