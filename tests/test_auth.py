import time

import jwt
import pytest

from agent_customer_support.auth import (
    MAX_PASSWORD_BYTES,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from agent_customer_support.config import get_settings


def test_hash_and_verify_roundtrip():
    hashed = hash_password("correct horse")
    assert hashed != "correct horse"
    assert verify_password("correct horse", hashed)
    assert not verify_password("wrong horse", hashed)


def test_hashing_is_salted():
    # Same password, different hashes — otherwise identical passwords are visible as
    # identical rows to anyone who reads the table.
    assert hash_password("same") != hash_password("same")


def test_verify_against_missing_hash_is_false():
    assert not verify_password("anything", None)


def test_verify_tolerates_garbage_hash():
    # A corrupted stored value must fail the login, not 500 the endpoint.
    assert not verify_password("anything", "not-a-bcrypt-hash")


def test_hash_rejects_overlong_password():
    with pytest.raises(ValueError):
        hash_password("x" * (MAX_PASSWORD_BYTES + 1))


def test_verify_tolerates_overlong_candidate():
    # bcrypt 5 raises above 72 bytes; an attacker posting a huge password must get a
    # failed login rather than an unhandled exception.
    assert not verify_password("x" * 500, hash_password("short"))


def test_token_roundtrip_carries_sub_and_role():
    token = create_access_token("cust1", "admin")
    payload = decode_access_token(token)
    assert payload["sub"] == "cust1"
    assert payload["role"] == "admin"


def test_token_signed_with_another_secret_is_rejected():
    s = get_settings()
    forged = jwt.encode(
        {"sub": "cust1", "role": "admin"}, "not-the-secret", algorithm=s.jwt_algorithm
    )
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(forged)


def test_expired_token_is_rejected():
    s = get_settings()
    expired = jwt.encode(
        {"sub": "cust1", "role": "user", "exp": int(time.time()) - 60},
        s.jwt_secret,
        algorithm=s.jwt_algorithm,
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(expired)


def test_unsigned_token_is_rejected():
    # alg=none is the classic JWT bypass; PyJWT must refuse it because we pin algorithms.
    unsigned = jwt.encode({"sub": "cust1", "role": "admin"}, key="", algorithm="none")
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(unsigned)
