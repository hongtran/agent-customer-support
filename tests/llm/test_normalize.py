from agent_customer_support.llm.normalize import (
    to_anthropic_content,
    to_openai_content,
)
from agent_customer_support.models import Attachment


def test_anthropic_text_only():
    blocks = to_anthropic_content("hello", [])
    assert blocks == [{"type": "text", "text": "hello"}]


def test_anthropic_with_image():
    att = Attachment(kind="image", media_type="image/png", data="QUJD")
    blocks = to_anthropic_content("see this", [att])
    assert blocks[0] == {"type": "text", "text": "see this"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


def test_openai_text_only():
    content = to_openai_content("hello", [])
    assert content == "hello"


def test_openai_with_image():
    att = Attachment(kind="image", media_type="image/jpeg", data="QUJD")
    content = to_openai_content("see this", [att])
    assert content[0] == {"type": "text", "text": "see this"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,QUJD"},
    }
