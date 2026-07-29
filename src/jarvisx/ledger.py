from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable, Mapping
from typing import Any


GENESIS_HASH = "0" * 64


def _canonical_json(value: Any) -> str:
    """Return a stable UTF-8 JSON representation for hashing and persistence."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class OmegaLedger:
    """Append-only, hash-chained execution journal.

    The ledger stores JSON-native values only. A clock can be injected for
    deterministic tests and replay fixtures; production callers default to
    ``time.time_ns``.
    """

    def __init__(self, clock_ns: Callable[[], int] | None = None) -> None:
        self.chain: list[dict[str, Any]] = []
        self._clock_ns = clock_ns or time.time_ns

    def log(self, state: Mapping[str, Any], opcode: int) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            raise TypeError("ledger state must be a mapping")

        previous_hash = self.chain[-1]["hash"] if self.chain else GENESIS_HASH
        body: dict[str, Any] = {
            "timestamp_ns": int(self._clock_ns()),
            "opcode": int(opcode),
            "state": dict(state),
            "previous_hash": previous_hash,
        }
        digest = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
        entry = {**body, "hash": digest}
        self.chain.append(entry)
        return dict(entry)

    def verify(self) -> bool:
        previous_hash = GENESIS_HASH
        for entry in self.chain:
            if not isinstance(entry, dict):
                return False
            if entry.get("previous_hash") != previous_hash:
                return False

            required = {"timestamp_ns", "opcode", "state", "previous_hash", "hash"}
            if set(entry) != required:
                return False

            body = {key: entry[key] for key in required if key != "hash"}
            expected = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
            if not hmac.compare_digest(str(entry["hash"]), expected):
                return False
            previous_hash = expected
        return True
