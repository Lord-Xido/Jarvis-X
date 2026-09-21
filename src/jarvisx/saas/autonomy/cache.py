"""Context-bound temporal cache for agent and digital-twin computations."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _key(parts: object) -> str:
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    tenant_id: str
    key: str
    value: Any
    valid_from_ns: int
    valid_until_ns: int
    state_version: int
    scope_hash: str


class TemporalSemanticCache:
    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, str], CacheEntry] = {}

    def put(
        self,
        *,
        tenant_id: str,
        semantic_key: object,
        scope: object,
        value: Any,
        state_version: int,
        ttl_seconds: int,
        now_ns: Optional[int] = None,
    ) -> CacheEntry:
        now = now_ns or time.time_ns()
        entry = CacheEntry(
            tenant_id=tenant_id,
            key=_key(semantic_key),
            value=value,
            valid_from_ns=now,
            valid_until_ns=now + ttl_seconds * 1_000_000_000,
            state_version=state_version,
            scope_hash=_key(scope),
        )
        self._entries[(tenant_id, entry.key)] = entry
        return entry

    def get(
        self,
        *,
        tenant_id: str,
        semantic_key: object,
        scope: object,
        state_version: int,
        now_ns: Optional[int] = None,
    ) -> Optional[Any]:
        now = now_ns or time.time_ns()
        entry = self._entries.get((tenant_id, _key(semantic_key)))
        if entry is None:
            return None
        valid = (
            entry.scope_hash == _key(scope)
            and entry.state_version == state_version
            and entry.valid_from_ns <= now <= entry.valid_until_ns
        )
        return entry.value if valid else None

    def invalidate_tenant(self, tenant_id: str) -> int:
        keys = [key for key in self._entries if key[0] == tenant_id]
        for key in keys:
            self._entries.pop(key, None)
        return len(keys)
