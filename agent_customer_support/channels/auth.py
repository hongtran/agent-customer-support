from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent_customer_support.auth import create_access_token, verify_password
from agent_customer_support.channels.deps import get_current_customer, get_customer_registry
from agent_customer_support.config import get_settings
from agent_customer_support.models import CustomerProfile, Role
from agent_customer_support.stores.customer_registry import CustomerRegistry

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    # user_name IS the customer_id — there is no separate username column, which is why
    # login is a direct get_item with no secondary index.
    user_name: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    customer_id: str
    role: Role
    expires_in: int  # seconds


class MeResponse(BaseModel):
    customer_id: str
    name: str
    role: Role
    enabled_applications: list[str]


@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    registry: CustomerRegistry = Depends(get_customer_registry),
) -> LoginResponse:
    profile = await registry.get(body.user_name)
    # One generic failure for every cause — unknown id, no credentials set, wrong
    # password. Telling them apart hands an attacker a list of real customer ids.
    # verify_password still runs (against a dummy hash) when profile is None so the
    # timing doesn't leak what the message won't.
    if not verify_password(body.password, profile.password_hash if profile else None):
        raise HTTPException(status_code=401, detail="invalid credentials")
    assert profile is not None  # verify_password returns False for a missing profile
    s = get_settings()
    return LoginResponse(
        access_token=create_access_token(profile.customer_id, profile.role),
        customer_id=profile.customer_id,
        role=profile.role,
        expires_in=s.jwt_expire_minutes * 60,
    )


@router.get("/me", response_model=MeResponse)
async def me(customer: CustomerProfile = Depends(get_current_customer)) -> MeResponse:
    return MeResponse(
        customer_id=customer.customer_id,
        name=customer.name,
        role=customer.role,
        enabled_applications=customer.enabled_applications,
    )
