"""Password hashing, signed access tokens, and role authorization."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Iterable

ROLE_LEVEL = {
    "viewer": 10,
    "consultant": 20,
    "auditor": 30,
    "operations_manager": 40,
    "finance_admin": 50,
    "tenant_owner": 80,
    "platform_admin": 100,
}


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str, salt: str = "") -> str:
    if len(password) < 10:
        raise ValueError("password must contain at least 10 characters")
    raw_salt = bytes.fromhex(salt) if salt else secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=raw_salt, n=2**14, r=8, p=1)
    return "scrypt$%s$%s" % (raw_salt.hex(), digest.hex())


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt, expected = stored.split("$", 2)
        if scheme != "scrypt":
            return False
        actual = hash_password(password, salt).split("$", 2)[2]
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    role: str


class TokenService:
    def __init__(self, secret: str = "", ttl_seconds: int = 3600) -> None:
        self.secret = (secret or os.getenv("DM_TOKEN_SECRET", "")).encode("utf-8")
        if len(self.secret) < 32:
            raise ValueError("DM_TOKEN_SECRET must contain at least 32 characters")
        self.ttl_seconds = int(ttl_seconds)

    def issue(self, principal: Principal) -> str:
        now = int(time.time())
        header = _b64(
            json.dumps({"alg": "HS256", "typ": "DMJWT"}, separators=(",", ":")).encode()
        )
        payload = _b64(
            json.dumps(
                {
                    "sub": principal.user_id,
                    "tenant_id": principal.tenant_id,
                    "role": principal.role,
                    "iat": now,
                    "exp": now + self.ttl_seconds,
                },
                separators=(",", ":"),
            ).encode()
        )
        signed = "%s.%s" % (header, payload)
        signature = _b64(
            hmac.new(self.secret, signed.encode(), hashlib.sha256).digest()
        )
        return "%s.%s" % (signed, signature)

    def verify(self, token: str) -> Principal:
        try:
            header, payload, signature = token.split(".")
            signed = "%s.%s" % (header, payload)
            expected = _b64(
                hmac.new(self.secret, signed.encode(), hashlib.sha256).digest()
            )
            if not hmac.compare_digest(signature, expected):
                raise ValueError("invalid token signature")
            claims = json.loads(_unb64(payload))
            if int(claims["exp"]) < int(time.time()):
                raise ValueError("token expired")
            role = str(claims["role"])
            if role not in ROLE_LEVEL:
                raise ValueError("unknown role")
            return Principal(str(claims["sub"]), str(claims["tenant_id"]), role)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid access token") from exc


def require_role(principal: Principal, allowed: Iterable[str]) -> None:
    allowed_set = set(allowed)
    if principal.role == "platform_admin":
        return
    if principal.role not in allowed_set:
        raise PermissionError("role %s is not authorized" % principal.role)
