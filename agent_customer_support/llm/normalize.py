from agent_customer_support.models import Attachment


def to_anthropic_content(text: str, attachments: list[Attachment]) -> list[dict]:
    blocks: list[dict] = [{"type": "text", "text": text}]
    for a in attachments:
        if a.kind == "image":
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": a.media_type,
                        "data": a.data,
                    },
                }
            )
    return blocks


def to_openai_content(text: str, attachments: list[Attachment]):
    if not attachments:
        return text
    content: list[dict] = [{"type": "text", "text": text}]
    for a in attachments:
        if a.kind == "image":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{a.media_type};base64,{a.data}"},
                }
            )
    return content
