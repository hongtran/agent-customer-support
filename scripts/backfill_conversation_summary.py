"""Write the summary fields onto conversations stored before they existed.

The admin conversation list reads the `customer_id-updated_at-index` GSI. That index is
sparse: a row with no `updated_at` is not in it, so every conversation written before the
summary fields were added is invisible in the admin UI until it gets a new turn. This
script rewrites those rows once with `Conversation.refresh_summary()` — the same function
the store calls on every write — so they appear.

    poetry run python scripts/backfill_conversation_summary.py            # dry run
    poetry run python scripts/backfill_conversation_summary.py --apply

Only rows missing `updated_at` are touched, so re-running is safe. Run it after the GSI
exists (the API creates it on start when DYNAMODB_AUTO_CREATE_TABLES is on).

Caveat: a row rewritten here while a live chat appends to the same conversation could lose
that turn (both are whole-item puts). Run it at a quiet time.
"""

import argparse
import asyncio

from agent_customer_support.config import get_settings
from agent_customer_support.models import Conversation
from agent_customer_support.stores.conversation_store import ConversationStore
from agent_customer_support.stores.dynamo import get_resource


async def main(apply: bool) -> None:
    table_name = get_settings().table_conversations
    store = ConversationStore()
    scanned = missing = 0
    kwargs: dict = {}
    async with get_resource() as ddb:
        table = await ddb.Table(table_name)
        while True:
            res = await table.scan(**kwargs)
            for item in res.get("Items", []):
                scanned += 1
                if item.get("updated_at"):
                    continue
                conv = Conversation.model_validate(item)
                # A row with no owner cannot be listed under anyone; leave it alone.
                if not conv.customer_id:
                    continue
                missing += 1
                if apply:
                    await store.put(conv)
            last = res.get("LastEvaluatedKey")
            if not last:
                break
            kwargs["ExclusiveStartKey"] = last
    verb = "updated" if apply else "would update"
    print(f"scanned {scanned} conversations, {verb} {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    asyncio.run(main(parser.parse_args().apply))
