from agent_customer_support.config import get_settings
from agent_customer_support.models import ContactInfo, Conversation, Turn
from agent_customer_support.stores.dynamo import ensure_table, get_resource


class ConversationStore:
    def __init__(self) -> None:
        self.table_name = get_settings().table_conversations

    async def init(self) -> None:
        await ensure_table(self.table_name, key="conversation_id")

    async def load(self, conversation_id: str) -> Conversation:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"conversation_id": conversation_id})
        item = res.get("Item")
        if not item:
            return Conversation(conversation_id=conversation_id, customer_id="")
        return Conversation.model_validate(item)

    async def append(self, conversation_id: str, customer_id: str, turn: Turn) -> None:
        conv = await self.load(conversation_id)
        if not conv.customer_id:
            conv.customer_id = customer_id
        conv.turns.append(turn)
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=conv.model_dump(mode="json"))

    async def set_contact(
        self, conversation_id: str, customer_id: str, contact: ContactInfo
    ) -> None:
        """Record the contact the user left after a handoff. Same load-modify-put as
        `append`, so a row that does not exist yet is created with its customer."""
        conv = await self.load(conversation_id)
        if not conv.customer_id:
            conv.customer_id = customer_id
        conv.contact = contact
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=conv.model_dump(mode="json"))
