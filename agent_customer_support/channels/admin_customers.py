from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent_customer_support.auth import MAX_PASSWORD_BYTES, hash_password
from agent_customer_support.channels.deps import (
    get_customer_registry,
    get_usage_store,
    require_admin,
)
from agent_customer_support.models import CustomerId, CustomerProfile, Role
from agent_customer_support.stores.customer_registry import CustomerExistsError, CustomerRegistry
from agent_customer_support.stores.usage_store import UsageStore

router = APIRouter(
    prefix="/admin/customers",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class CustomerOut(BaseModel):
    """The shape customers are returned in. Built field-by-field rather than by
    dumping CustomerProfile, so password_hash cannot leak by accident."""

    customer_id: str
    name: str
    role: Role
    enabled_applications: list[str]
    config_notes: str | None = None
    has_password: bool
    daily_question_limit: int | None = None
    questions_used_today: int = 0

    @classmethod
    def of(cls, p: CustomerProfile, used_today: int = 0) -> "CustomerOut":
        return cls(
            customer_id=p.customer_id,
            name=p.name,
            role=p.role,
            enabled_applications=p.enabled_applications,
            config_notes=p.config_notes,
            has_password=p.password_hash is not None,
            daily_question_limit=p.daily_question_limit,
            questions_used_today=used_today,
        )


class CustomerCreate(BaseModel):
    customer_id: CustomerId  # the login username; permanent
    name: str
    password: str = Field(min_length=8, max_length=MAX_PASSWORD_BYTES)
    role: Role = "user"
    enabled_applications: list[str] = Field(default_factory=list)
    config_notes: str | None = None
    daily_question_limit: int | None = Field(default=None, ge=0)  # None = unlimited


class CustomerPatch(BaseModel):
    name: str | None = None
    password: str | None = Field(default=None, min_length=8, max_length=MAX_PASSWORD_BYTES)
    role: Role | None = None
    enabled_applications: list[str] | None = None
    config_notes: str | None = None
    # None is ambiguous here ("unchanged" vs "unlimited"), so the handler checks
    # model_fields_set: an explicit null clears the limit, an absent field keeps it.
    daily_question_limit: int | None = Field(default=None, ge=0)


async def _used_today(usage: UsageStore, p: CustomerProfile) -> int:
    # Only read the counter when a limit exists — an unlimited customer's count is
    # never shown, and this keeps the list at one DynamoDB read per limited customer.
    return await usage.get_today(p.customer_id) if p.daily_question_limit is not None else 0


@router.get("")
async def list_customers(
    registry: CustomerRegistry = Depends(get_customer_registry),
    usage: UsageStore = Depends(get_usage_store),
) -> list[CustomerOut]:
    return [CustomerOut.of(p, await _used_today(usage, p)) for p in await registry.list()]


@router.post("", status_code=201)
async def create_customer(
    body: CustomerCreate,
    registry: CustomerRegistry = Depends(get_customer_registry),
) -> CustomerOut:
    profile = CustomerProfile(
        customer_id=body.customer_id,
        name=body.name,
        role=body.role,
        enabled_applications=body.enabled_applications,
        config_notes=body.config_notes,
        daily_question_limit=body.daily_question_limit,
        password_hash=hash_password(body.password),
    )
    try:
        await registry.create(profile)
    except CustomerExistsError as exc:
        raise HTTPException(
            status_code=409, detail=f"customer_id {body.customer_id!r} already exists"
        ) from exc
    return CustomerOut.of(profile)


@router.patch("/{customer_id}")
async def update_customer(
    customer_id: str,
    patch: CustomerPatch,
    registry: CustomerRegistry = Depends(get_customer_registry),
    usage: UsageStore = Depends(get_usage_store),
) -> CustomerOut:
    profile = await registry.get(customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="not found")
    if patch.name is not None:
        profile.name = patch.name
    if patch.role is not None:
        profile.role = patch.role
    if patch.enabled_applications is not None:
        profile.enabled_applications = patch.enabled_applications
    if patch.config_notes is not None:
        profile.config_notes = patch.config_notes
    if "daily_question_limit" in patch.model_fields_set:
        profile.daily_question_limit = patch.daily_question_limit
    # An absent password leaves the existing hash alone; only an explicit one re-hashes,
    # so a rename can't wipe someone's credentials.
    if patch.password is not None:
        profile.password_hash = hash_password(patch.password)
    await registry.put(profile)
    return CustomerOut.of(profile, await _used_today(usage, profile))
