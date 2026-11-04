import logging
import uuid
from datetime import UTC, date, datetime
from typing import Any

from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

from agent_customer_support.config import get_settings
from agent_customer_support.models import (
    RESOLVED_STATUSES,
    ContactInfo,
    FeatureCategory,
    IssueSummary,
    RequestFile,
    RequestRecord,
    Requester,
    RequestStatus,
    RequestType,
    StatusChange,
)
from agent_customer_support.stores.dynamo import get_resource

logger = logging.getLogger(__name__)

# Lists one customer's rows, newest first. Every issue list (the customer's own page and
# the admin's per-customer tab) is a query on this index; nothing scans the table, which
# would read and bill every tenant's transcripts on each page load.
BY_CUSTOMER_INDEX = "customer_id-created_at-index"

_INDEX_ATTRS = [
    {"AttributeName": "customer_id", "AttributeType": "S"},
    {"AttributeName": "created_at", "AttributeType": "S"},
]
_INDEX = {
    "IndexName": BY_CUSTOMER_INDEX,
    "KeySchema": [
        {"AttributeName": "customer_id", "KeyType": "HASH"},
        {"AttributeName": "created_at", "KeyType": "RANGE"},
    ],
    # What a list row shows, plus `type` to filter on. Not ALL: the transcript is the
    # bulk of a row and a list never shows it.
    "Projection": {
        "ProjectionType": "INCLUDE",
        "NonKeyAttributes": [
            "type",
            "title",
            "summary",
            "application",
            "status",
            "status_updated_at",
            "mantis_issue_id",
            "mantis_issue_url",
        ],
    },
}


# The most keys one BatchGetItem call accepts.
_BATCH_GET_MAX = 100


def _key_bound(t: datetime) -> str:
    """A `created_at` range bound, compared as a string against the stored ISO value.

    Stored values look like `2026-11-03T17:00:00.123456Z`. The bound is written to the
    second, with no fraction and no `Z`: a string sorts before every longer string it
    is a prefix of, so a bound is lower than every value stored within that second. That
    makes `>= bound` include the whole second and `< bound` exclude it, whatever
    fraction or offset suffix a stored value carries.
    """
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")


class _Unset:
    """Type of `UNSET`: lets `update_issue` tell "not sent" from "clear it" (None)."""


UNSET = _Unset()


class RequestBacklog:
    def __init__(self) -> None:
        self.table_name = get_settings().table_requests

    async def init(self) -> None:
        """Create the table with its by-customer index, or add the index to a table that
        predates it. Same shape as `ConversationStore.init`: with auto-create off (prod),
        the table and the index are infrastructure's job."""
        if not get_settings().dynamodb_auto_create_tables:
            return
        async with get_resource() as ddb:
            client = ddb.meta.client
            try:
                await client.create_table(
                    TableName=self.table_name,
                    KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
                    AttributeDefinitions=[
                        {"AttributeName": "id", "AttributeType": "S"},
                        *_INDEX_ATTRS,
                    ],
                    GlobalSecondaryIndexes=[_INDEX],
                    BillingMode="PAY_PER_REQUEST",
                )
                return
            except ClientError as e:
                if e.response["Error"]["Code"] != "ResourceInUseException":
                    raise
            desc = await client.describe_table(TableName=self.table_name)
            existing = {i["IndexName"] for i in desc["Table"].get("GlobalSecondaryIndexes", [])}
            if BY_CUSTOMER_INDEX in existing:
                return
            await client.update_table(
                TableName=self.table_name,
                AttributeDefinitions=_INDEX_ATTRS,
                GlobalSecondaryIndexUpdates=[{"Create": _INDEX}],
            )

    async def add(
        self,
        *,
        customer_id: str,
        type: RequestType,
        summary: str,
        application: str | None = None,
        transcript: str = "",
        title: str | None = None,
        mantis_issue_id: int | None = None,
        mantis_issue_url: str | None = None,
        id: str | None = None,
        status: RequestStatus = "pending",
        requester: Requester | None = None,
        reason: str | None = None,
        files: list[RequestFile] | None = None,
    ) -> RequestRecord:
        """Write a new row. `id` may be given when files were stored under it first
        (a feature request uploads to `requests/<id>/` before the row exists)."""
        rec = RequestRecord(
            id=id or str(uuid.uuid4()),
            status=status,
            requester=requester,
            reason=reason,
            files=files or [],
            customer_id=customer_id,
            type=type,
            summary=summary,
            title=title,
            application=application,
            transcript=transcript,
            mantis_issue_id=mantis_issue_id,
            mantis_issue_url=mantis_issue_url,
        )
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.put_item(Item=rec.model_dump(mode="json"))
        return rec

    async def set_contact(self, request_id: str, contact: ContactInfo) -> None:
        """Attach the contact the user gave after the handoff. A single-attribute
        update, not a read-modify-write of the row, so it cannot roll back anything
        written in between."""
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            await table.update_item(
                Key={"id": request_id},
                UpdateExpression="SET contact = :c",
                ExpressionAttributeValues={":c": contact.model_dump(mode="json")},
            )

    async def get(self, request_id: str) -> RequestRecord | None:
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            res = await table.get_item(Key={"id": request_id})
        item = res.get("Item")
        return RequestRecord.model_validate(item) if item else None

    async def list_bugs(self, customer_id: str) -> list[IssueSummary]:
        """One customer's verified bugs, newest first."""
        return [IssueSummary.model_validate(i) for i in await self.list_by_type(customer_id, "bug")]

    async def list_by_type(
        self,
        customer_id: str,
        type: RequestType,
        *,
        created_from: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """One customer's rows of one type, newest first, as the raw GSI items (only the
        projected fields). Each caller validates them into its own summary model.

        `created_from` (inclusive) and `created_before` (exclusive) limit `created_at`.
        That is a condition on the index sort key, so DynamoDB reads only the rows in
        the range instead of filtering them after the read.

        Follows `LastEvaluatedKey`: a query page stops at 1 MB, and the `type` filter is
        applied after that limit, so one page can come back short or even empty.
        """
        key = Key("customer_id").eq(customer_id)
        lo = _key_bound(created_from) if created_from else None
        hi = _key_bound(created_before) if created_before else None
        if lo and hi:
            # BETWEEN is inclusive, but no stored value equals `hi` (see `_key_bound`),
            # so this is lo <= created_at < hi.
            key = key & Key("created_at").between(lo, hi)
        elif lo:
            key = key & Key("created_at").gte(lo)
        elif hi:
            key = key & Key("created_at").lt(hi)
        kwargs: dict[str, Any] = {
            "IndexName": BY_CUSTOMER_INDEX,
            "KeyConditionExpression": key,
            "FilterExpression": Attr("type").eq(type),
            "ScanIndexForward": False,
        }
        items: list[dict[str, Any]] = []
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            while True:
                res = await table.query(**kwargs)
                items.extend(res.get("Items", []))
                last = res.get("LastEvaluatedKey")
                if not last:
                    break
                kwargs["ExclusiveStartKey"] = last
        return items

    async def get_many(self, ids: list[str]) -> list[RequestRecord]:
        """Full rows for these ids, in the same order, without the transcript (the bulk
        of a bug row, and nothing an export shows). For the fields the index does not
        project — note, due date, requester, category… — without one `get_item` each.

        `BatchGetItem` takes at most 100 keys per call and may hand some back as
        `UnprocessedKeys` under load; those are asked again. A missing id is left out.
        A row that does not validate is skipped and logged, like the list endpoints do.
        """
        fields = [f for f in RequestRecord.model_fields if f != "transcript"]
        names = {f"#f{i}": f for i, f in enumerate(fields)}
        projection = ", ".join(names)
        found: dict[str, dict[str, Any]] = {}
        async with get_resource() as ddb:
            for start in range(0, len(ids), _BATCH_GET_MAX):
                chunk = list(dict.fromkeys(ids[start : start + _BATCH_GET_MAX]))
                request: dict[str, Any] = {
                    self.table_name: {
                        "Keys": [{"id": i} for i in chunk],
                        "ProjectionExpression": projection,
                        "ExpressionAttributeNames": names,
                    }
                }
                while request:
                    res = await ddb.batch_get_item(RequestItems=request)
                    for item in res.get("Responses", {}).get(self.table_name, []):
                        found[item["id"]] = item
                    request = res.get("UnprocessedKeys") or {}
        records: list[RequestRecord] = []
        for i in ids:
            item = found.get(i)
            if item is None:
                continue
            try:
                records.append(RequestRecord.model_validate(item))
            except ValueError as exc:
                logger.warning("skipping invalid request row %s: %s", i, exc)
        return records

    async def update_issue(
        self,
        request_id: str,
        customer_id: str,
        *,
        by: str,
        type: RequestType = "bug",
        status: RequestStatus | None = None,
        note: str | None | _Unset = UNSET,
        due_date: date | None | _Unset = UNSET,
        category: FeatureCategory | None | _Unset = UNSET,
    ) -> RequestRecord | None:
        """Apply an admin's edit to a bug or a feature request (`type`). Returns the
        updated row, or None when there is no row of that type with this id for this
        customer. The caller limits `status` to the values valid for `type`.

        Only what is passed changes: `UNSET` leaves a field alone, `None` (or a blank
        note) removes it. A status change also appends to `status_history` and sets or
        removes `resolved_at`.

        One `update_item`, so the history append runs on the server: two admins saving at
        once both land in it, where a read-modify-write would drop one — the same reason
        `set_contact` is a single update. The ownership and type checks are in the
        condition, so a wrong id never creates a stub row or touches another customer's
        request.
        """
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        sets: list[str] = []
        removes: list[str] = []
        names: dict[str, str] = {}
        values: dict[str, Any] = {}

        # A blank note means "clear it", same as None.
        clean_note = note.strip() or None if isinstance(note, str) else None

        if status is not None:
            names |= {"#s": "status", "#h": "status_history", "#r": "resolved_at"}
            entry = StatusChange.model_validate(
                {"status": status, "at": now, "by": by, "note": clean_note}
            )
            values |= {
                ":s": status,
                ":t": now,
                ":b": by,
                ":entry": [entry.model_dump(mode="json")],
                ":empty": [],
            }
            sets += [
                "#s = :s",
                "status_updated_at = :t",
                "status_updated_by = :b",
                "#h = list_append(if_not_exists(#h, :empty), :entry)",
            ]
            if status in RESOLVED_STATUSES:
                sets.append("#r = :t")
            else:
                removes.append("#r")

        if not isinstance(note, _Unset):
            names["#n"] = "note"
            if clean_note:
                sets.append("#n = :n")
                values[":n"] = clean_note
            else:
                removes.append("#n")

        if not isinstance(due_date, _Unset):
            names["#d"] = "due_date"
            if due_date is not None:
                sets.append("#d = :d")
                values[":d"] = due_date.isoformat()
            else:
                removes.append("#d")

        if not isinstance(category, _Unset):
            names["#c"] = "category"
            if category is not None:
                sets.append("#c = :c")
                values[":c"] = category
            else:
                removes.append("#c")

        if not sets and not removes:
            raise ValueError("update_issue called with nothing to change")
        expr = " ".join(
            part
            for part in (
                f"SET {', '.join(sets)}" if sets else "",
                f"REMOVE {', '.join(removes)}" if removes else "",
            )
            if part
        )
        kwargs: dict[str, Any] = {
            "Key": {"id": request_id},
            "UpdateExpression": expr,
            "ConditionExpression": (
                Attr("id").exists() & Attr("customer_id").eq(customer_id) & Attr("type").eq(type)
            ),
            "ExpressionAttributeNames": names,
            "ReturnValues": "ALL_NEW",
        }
        # DynamoDB rejects an empty value map, which a remove-only update would send.
        if values:
            kwargs["ExpressionAttributeValues"] = values
        async with get_resource() as ddb:
            table = await ddb.Table(self.table_name)
            try:
                res = await table.update_item(**kwargs)
            except ClientError as e:
                if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                    return None
                raise
        return RequestRecord.model_validate(res["Attributes"])
