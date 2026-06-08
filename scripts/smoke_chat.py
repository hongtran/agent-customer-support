import asyncio
from agent_customer_support.models import CustomerProfile
from agent_customer_support.stores.customer_registry import CustomerRegistry
from agent_customer_support.agent.core import AgentCore


async def main() -> None:
    reg = CustomerRegistry(); await reg.init()
    await reg.put(CustomerProfile(
        customer_id="ttp", name="TTP", enabled_modules=["yeu-cau-thu-nghiem", "xet-nghiem"]
    ))
    agent = AgentCore()
    for msg in ["Làm sao xử lý PYC sự cố?", "tôi không thấy phiếu nào cả"]:
        reply = await agent.handle_turn(customer_id="ttp", conversation_id="smoke1", user_msg=msg)
        print(f"\nUSER: {msg}\nAGENT: {reply.reply}\n(escalated={reply.escalated})")


if __name__ == "__main__":
    asyncio.run(main())
