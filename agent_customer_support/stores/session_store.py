from redis.asyncio import Redis
from agent_customer_support.config import get_settings
from agent_customer_support.models import SessionState


class SessionStore:
    def __init__(self, client: Redis | None = None) -> None:
        s = get_settings()
        self.ttl = s.session_ttl_seconds
        self.client = client or Redis.from_url(s.redis_url)

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"acs:session:{conversation_id}"

    async def get(self, conversation_id: str) -> SessionState:
        raw = await self.client.get(self._key(conversation_id))
        if raw is None:
            return SessionState(conversation_id=conversation_id)
        return SessionState.model_validate_json(raw)

    async def save(self, state: SessionState) -> None:
        await self.client.set(
            self._key(state.conversation_id),
            state.model_dump_json(),
            ex=self.ttl,
        )
