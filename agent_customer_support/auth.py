"""Password hashing and access-token minting.

The only module that imports bcrypt or PyJWT. Everything else goes through these
four functions, so the choice of algorithm stays swappable in one place.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from agent_customer_support.config import get_settings

# bcrypt hashes at most 72 bytes and bcrypt 5.x *raises* on anything longer rather
# than silently truncating. Enforced on the way in so a long password is a 422 at
# the API boundary, never a 500 from inside the hasher.
MAX_PASSWORD_BYTES = 72

# A real hash of a value nobody knows, used to burn the same ~100ms of bcrypt work
# when an account is missing or has no password set. Without it, "unknown customer"
# returns noticeably faster than "wrong password" and login timing turns into a
# customer-id oracle.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password-for-constant-time-compare", bcrypt.gensalt())


def hash_password(plain: str) -> str:
    encoded = plain.encode()
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(f"password must be at most {MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str | None) -> bool:
    """Check a password against a stored hash.

    Never raises: a malformed stored hash, or an over-long candidate password, is a
    failed login rather than a 500. `hashed=None` (a profile with no credentials set)
    still pays the full bcrypt cost — see _DUMMY_HASH.
    """
    encoded = plain.encode()[:MAX_PASSWORD_BYTES]
    if hashed is None:
        bcrypt.checkpw(encoded, _DUMMY_HASH)
        return False
    try:
        return bcrypt.checkpw(encoded, hashed.encode())
    except ValueError:
        return False


def _secret() -> str:
    secret = get_settings().jwt_secret
    if not secret:
        raise RuntimeError("JWT_SECRET is not configured")
    return secret


def create_access_token(customer_id: str, role: str) -> str:
    s = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": customer_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=s.jwt_expire_minutes),
    }
    return jwt.encode(payload, _secret(), algorithm=s.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and verify a token. Raises jwt.PyJWTError on anything invalid."""
    return jwt.decode(token, _secret(), algorithms=[get_settings().jwt_algorithm])
