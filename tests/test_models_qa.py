from agent_customer_support.models import QARecord, Turn, ChatResponse
from agent_customer_support.config import Settings


def test_qarecord_defaults():
    rec = QARecord(question="làm sao đổi mật khẩu?", source="manual")
    assert rec.id  # auto uuid
    assert "-" in rec.id  # canonical uuid form (Qdrant-compatible)
    assert rec.status == "pending"
    assert rec.answer == ""
    assert rec.application is None


def test_turn_has_auto_id_and_chatresponse_message_id():
    t = Turn(role="assistant", content="hi")
    assert t.id
    resp = ChatResponse(conversation_id="c1", reply="hi", message_id=t.id)
    assert resp.message_id == t.id


def test_qa_settings_present():
    s = Settings()
    assert s.table_qa == "acs_qa"
    assert s.qa_collection  # from QA_COLLECTION env stub
    assert s.jwt_secret  # from JWT_SECRET env stub
