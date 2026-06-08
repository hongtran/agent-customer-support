import asyncio, json, sys
from agent_customer_support.agent.core import AgentCore


async def main(golden_path: str, customer_id: str = "ttp") -> None:
    items = json.loads(open(golden_path).read())
    agent = AgentCore()
    correct_class = deflected = 0
    for i, it in enumerate(items):
        reply = await agent.handle_turn(
            customer_id=customer_id, conversation_id=f"eval-{i}", user_msg=it["request"]
        )
        pred = "feature" if reply.escalated or "chuyển" in reply.reply.lower() else "how_to"
        correct_class += int(pred == it["expected_class"])
        deflected += int(pred == "how_to" and not reply.escalated)
    n = len(items)
    print(f"triage accuracy: {correct_class}/{n} = {correct_class/n*100:.0f}%")
    print(f"deflected (tự trả lời): {deflected}/{n}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "eval/golden.json"))
