"""Authenticated local process boundary for isolated ADR-016 shadow participants.

Each service process owns ONE SQLite participant database and an AF_UNIX socket.
The coordinator has its own separate SQLite database and communicates with the
participants through versioned HMAC-SHA256 authenticated RPC messages.

Scope: single-host, trusted local demonstration. Not mTLS, network partition
handling, live VM/OS/ANN RPC, host-administrator protection, or distributed ACID.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import sqlite3
import time

from reference.canonical_shadow_tx.bridge import DOMAINS, DomainFrame, encoded
from reference.canonical_shadow_tx.multiprocess import Coordinator, Participant
from reference.canonical_shadow_tx.decision_authority import DecisionAuthority

PROTOCOL = "CanonicalShadowIPC/1"
MAX_MESSAGE = 1_048_576
MAX_SKEW_SECONDS = 45
ENV_SECRET = "JARVISX_SHADOW_IPC_SECRET"


def _secret() -> bytes:
    raw = os.environ.get(ENV_SECRET, "")
    if not 32 <= len(raw) <= 256 or not raw.isascii():
        raise RuntimeError("a 32-256 character ASCII local IPC secret is required")
    return raw.encode("ascii")


def _mac(secret: bytes, obj: dict) -> str:
    return hmac.new(secret, encoded(obj), hashlib.sha256).hexdigest()


def _frame_wire(frame: DomainFrame) -> dict:
    return {"domain": frame.domain, "profile": frame.profile,
            "payload_hex": frame.payload.hex()}


def _parse_frame(obj: dict) -> DomainFrame:
    if type(obj) is not dict or set(obj) != {"domain", "profile", "payload_hex"}:
        raise ValueError("invalid frame")
    return DomainFrame(obj["domain"], obj["profile"],
                       bytes.fromhex(obj["payload_hex"]))


def make_request(secret: bytes, domain: str, action: str, args: dict,
                 *, nonce: str | None = None, timestamp: int | None = None) -> dict:
    payload = {
        "protocol": PROTOCOL, "domain": domain, "action": action, "args": args,
        "nonce": nonce if nonce is not None else secrets.token_hex(16),
        "timestamp": int(time.time()) if timestamp is None else timestamp,
    }
    return {**payload, "mac": _mac(secret, payload)}


class AuthenticationError(RuntimeError):
    pass


class PeerClient:
    """Signed requests and signed responses; no credentials in URLs or logs."""

    def __init__(self, root: Path, domain: str, secret: bytes, timeout: float = 3.0):
        if domain not in DOMAINS:
            raise ValueError("unknown domain")
        self.socket_path = root / (domain + ".sock")
        self.domain = domain
        self.secret = secret
        self.timeout = timeout

    def exchange(self, message: dict) -> dict:
        body = encoded(message) + b"\n"
        if len(body) > MAX_MESSAGE:
            raise ValueError("RPC message exceeds limit")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(str(self.socket_path))
            connection.sendall(body)
            packet = bytearray()
            while not packet.endswith(b"\n"):
                data = connection.recv(min(8192, MAX_MESSAGE + 1 - len(packet)))
                if not data or len(packet) + len(data) > MAX_MESSAGE:
                    raise AuthenticationError("invalid or oversized RPC response")
                packet.extend(data)
        result = json.loads(bytes(packet))
        if not isinstance(result, dict) or set(result) != {"body", "mac"}:
            raise AuthenticationError("invalid signed response")
        payload = result["body"]
        if not isinstance(payload, dict) or not hmac.compare_digest(
            str(result["mac"]), _mac(self.secret, payload)
        ):
            raise AuthenticationError("RPC response authentication failed")
        if payload.get("nonce") != message.get("nonce") or payload.get("domain") != self.domain:
            raise AuthenticationError("RPC response identity mismatch")
        if payload.get("ok") is not True:
            raise RuntimeError("participant rejected authenticated " +
                               str(message.get("action")) + " request")
        return payload["result"]

    def call(self, action: str, args: dict | None = None) -> dict:
        return self.exchange(make_request(self.secret, self.domain, action, args or {}))


class RemoteParticipant:
    """Coordinator-facing adapter: actual participant code executes remotely."""

    def __init__(self, peer: PeerClient):
        self.peer = peer

    def __enter__(self) -> "RemoteParticipant":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def prepare(self, txid: str, epoch: int, token: int, frame: DomainFrame) -> None:
        self.peer.call("prepare", {"txid": txid, "epoch": epoch, "token": token,
                                   "frame": _frame_wire(frame)})

    def matches_prepared(self, txid: str, epoch: int, token: int,
                         frame: DomainFrame) -> bool:
        return self.peer.call("matches_prepared", {
            "txid": txid, "epoch": epoch, "token": token,
            "frame": _frame_wire(frame),
        })["matches"]

    def commit(self, txid: str, token: int) -> None:
        self.peer.call("commit", {"txid": txid, "token": token})

    def abort(self, txid: str) -> None:
        self.peer.call("abort", {"txid": txid})

    def state(self) -> tuple[int, int, DomainFrame | None]:
        obj = self.peer.call("state")
        frame = None if obj["frame"] is None else _parse_frame(obj["frame"])
        return obj["version"], obj["fence"], frame

    def verify(self) -> bool:
        return self.peer.call("verify")["verified"]

    def status(self, txid: str) -> str | None:
        return self.peer.call("status", {"txid": txid})["status"]


class IPCCoordinator(Coordinator):
    """Keep coordinator WAL, decisions and recovery; replace direct DB access."""

    def __init__(self, root: str | Path, *, secret: bytes | None = None):
        self.rpc_secret = _secret() if secret is None else secret
        if len(self.rpc_secret) < 32:
            raise ValueError("IPC shared secret too short")
        super().__init__(root)

    def _participant(self, domain: str, *, create: bool = False) -> RemoteParticipant:
        # All local SQLite participant writes remain inside separate daemon PIDs.
        # Do not silently fall back to direct DB access on RPC failure.
        if domain not in DOMAINS:
            raise ValueError("unknown domain")
        if not create and not (self.root / (domain + ".sqlite3")).is_file():
            raise RuntimeError("required participant storage is missing")
        return RemoteParticipant(PeerClient(self.root, domain, self.rpc_secret))


class ParticipantService:
    """Single-threaded AF_UNIX server; durable nonce replay register per DB."""

    def __init__(self, root: Path, domain: str, secret: bytes):
        if domain not in DOMAINS or len(secret) < 32:
            raise ValueError("invalid participant identity or IPC secret")
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.domain, self.secret = root, domain, secret
        self.participant = Participant(root, domain)
        self.participant.db.execute(
            "CREATE TABLE IF NOT EXISTS rpc_nonces "
            "(nonce TEXT PRIMARY KEY, timestamp INTEGER NOT NULL)"
        )

    def close(self) -> None:
        self.participant.close()

    def _authenticate(self, message: object) -> None:
        if not isinstance(message, dict) or set(message) != {
            "protocol", "domain", "action", "args", "nonce", "timestamp", "mac"
        }:
            raise AuthenticationError("invalid RPC schema")
        if (message["protocol"] != PROTOCOL or message["domain"] != self.domain
                or type(message["nonce"]) is not str
                or len(message["nonce"]) != 32
                or any(x not in "0123456789abcdef" for x in message["nonce"])
                or type(message["timestamp"]) is not int
                or abs(int(time.time()) - message["timestamp"]) > MAX_SKEW_SECONDS
                or type(message["action"]) is not str
                or type(message["args"]) is not dict
                or type(message["mac"]) is not str):
            raise AuthenticationError("invalid authenticated RPC fields")
        unsigned = {key: value for key, value in message.items() if key != "mac"}
        if not hmac.compare_digest(message["mac"], _mac(self.secret, unsigned)):
            raise AuthenticationError("invalid RPC MAC")
        db = self.participant.db
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM rpc_nonces WHERE timestamp < ?",
                       (int(time.time()) - 2 * MAX_SKEW_SECONDS,))
            db.execute(
                "INSERT INTO rpc_nonces(nonce,timestamp) VALUES(?,?)",
                (message["nonce"], message["timestamp"]),
            )
            db.execute("COMMIT")
        except BaseException as exc:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise AuthenticationError("replayed or unrecordable RPC nonce") from exc

    def dispatch(self, message: object) -> dict:
        self._authenticate(message)
        assert isinstance(message, dict)
        action, args = message["action"], message["args"]
        p = self.participant
        authority = DecisionAuthority(self.root, self.domain)
        if action == "state" and not args:
            version, fence, frame = p.state()
            return {"version": version, "fence": fence,
                    "frame": None if frame is None else _frame_wire(frame)}
        if action == "verify" and not args:
            return {"verified": p.verify()}
        if action == "prepare" and set(args) == {"txid", "epoch", "token", "frame"}:
            frame = _parse_frame(args["frame"])
            authority.check_prepare(args["txid"], args["epoch"], args["token"], frame)
            p.prepare(args["txid"], args["epoch"], args["token"], frame)
            return {"prepared": True}
        if action == "matches_prepared" and set(args) == {
                "txid", "epoch", "token", "frame"}:
            return {"matches": p.matches_prepared(
                args["txid"], args["epoch"], args["token"], _parse_frame(args["frame"])
            )}
        if action == "commit" and set(args) == {"txid", "token"}:
            authority.check_commit(args["txid"], args["token"])
            p.commit(args["txid"], args["token"])
            return {"committed": True}
        if action == "abort" and set(args) == {"txid"}:
            authority.check_abort(args["txid"])
            p.abort(args["txid"])
            return {"aborted": True}
        if action == "status" and set(args) == {"txid"}:
            return {"status": p.status(args["txid"])}
        raise ValueError("unsupported RPC action or parameters")


def serve(root: Path, domain: str) -> None:
    """Local only; do not expose socket via shared untrusted filesystem."""
    secret = _secret()
    if domain not in DOMAINS:
        raise ValueError("invalid domain")
    os.umask(0o077)
    root = root.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root / (domain + ".sock")
    if target.exists() or target.is_symlink():
        raise RuntimeError("existing socket path; refusing silent unlink")
    service = ParticipantService(root, domain, secret)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            self.request.settimeout(3.0)
            try:
                message = self.rfile.readline(MAX_MESSAGE + 1)
                if not message.endswith(b"\n") or len(message) > MAX_MESSAGE:
                    raise ValueError("oversized RPC request")
                request = json.loads(message)
                result = service.dispatch(request)
                response = {"nonce": request.get("nonce"), "domain": domain,
                            "ok": True, "result": result}
            except Exception:
                # Do not return unsanitised stack traces or secret material.
                nonce = request.get("nonce") if isinstance(locals().get("request"), dict) else None
                response = {"nonce": nonce, "domain": domain, "ok": False,
                            "result": {"rejected": True}}
            wire = {"body": response, "mac": _mac(secret, response)}
            self.wfile.write(encoded(wire) + b"\n")

    try:
        with socketserver.UnixStreamServer(str(target), Handler) as server:
            os.chmod(target, 0o600)
            server.serve_forever(poll_interval=0.05)
    finally:
        service.close()
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("command", choices=("serve", "recover", "status"))
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--txid", default="demo-001")
    args = parser.parse_args()
    if args.command == "serve":
        if args.domain is None:
            parser.error("serve requires --domain")
        serve(args.root, args.domain)
    else:
        with IPCCoordinator(args.root) as coordinator:
            output = coordinator.recover() if args.command == "recover" else (
                coordinator.receipt(args.txid)
            )
        print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
