import logging
import httpx
from agent_customer_support.config import get_settings

logger = logging.getLogger(__name__)


class Escalator:
    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = (
            webhook_url if webhook_url is not None else get_settings().zalo_cs_webhook_url
        )

    async def escalate(self, *, customer_id: str, reason: str, transcript: str) -> None:
        if not self.webhook_url:
            logger.warning(
                "No Zalo CS webhook configured; escalation logged only: %s/%s", customer_id, reason
            )
            return
        payload = {
            "text": f"[HỖ TRỢ] Khách {customer_id}\nLý do: {reason}\n---\n{transcript[:3000]}"
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(self.webhook_url, json=payload)
            resp.raise_for_status()
