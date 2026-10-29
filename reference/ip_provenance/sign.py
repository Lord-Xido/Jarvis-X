"""Optional owner-key Ed25519 signature over an already pinned IP baseline.

No key material is generated, stored or supplied by this repository. A valid
signature proves control of the provided private key only when the public key
identity has been independently established by the owner.

Signing accepts an owner-supplied PEM key and a *fully reproduced* anchored
manifest. Verification requires a separate trusted public PEM key; a public key
embedded in the signed envelope is never trusted as the identity source.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)

from reference.ip_provenance.freeze import (
    FROZEN_COMMIT, FROZEN_GIT_TREE,
    canonical_bytes, capture, verify_against_anchor, verify_manifest_digest,
)

SCHEMA = "JarvisX/IPFreeze/Ed25519DetachedSignature1"
CONTEXT = b"JARVISX-IPFREEZE/v1/manifest-owner-signature\x00"


def _digest(manifest_bytes: bytes) -> bytes:
    return hashlib.sha256(CONTEXT + hashlib.sha256(manifest_bytes).digest()).digest()


def _public_fingerprint(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _verified_manifest(path: Path) -> bytes:
    """No signature over unverified or silently changed repo source."""
    expected = capture()
    verify_against_anchor(expected)
    raw = path.read_bytes()
    if raw != canonical_bytes(expected):
        raise ValueError("manifest is not the exact anchored frozen source inventory")
    verify_manifest_digest(raw)
    return raw


def sign_detached(manifest_path: Path, private_key_pem_path: Path) -> dict:
    raw = _verified_manifest(manifest_path)
    key = serialization.load_pem_private_key(
        private_key_pem_path.read_bytes(), password=None,
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("owner key must be PEM Ed25519 private key")
    signature = key.sign(_digest(raw))
    return {
        "schema": SCHEMA,
        "algorithm": "Ed25519",
        "content_binding": "SHA256(context || SHA256(canonical_manifest))",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "frozen_git_commit": FROZEN_COMMIT,
        "frozen_git_tree": FROZEN_GIT_TREE,
        "public_key_sha256": _public_fingerprint(key.public_key()),
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "identity_notice": (
            "Fingerprint requires independent owner identity verification; "
            "a valid key signature is not a trusted timestamp or patent right"
        ),
    }


def verify_detached(manifest_path: Path, signature_path: Path,
                    trusted_public_pem: Path) -> bool:
    raw = _verified_manifest(manifest_path)
    envelope = json.loads(signature_path.read_text("utf-8"))
    expected_keys = {
        "schema", "algorithm", "content_binding", "manifest_sha256",
        "frozen_git_commit", "frozen_git_tree", "public_key_sha256",
        "signature_base64", "identity_notice",
    }
    if type(envelope) is not dict or set(envelope) != expected_keys:
        raise ValueError("unknown signature schema")
    if (envelope["schema"] != SCHEMA or envelope["algorithm"] != "Ed25519"
            or envelope["frozen_git_commit"] != FROZEN_COMMIT
            or envelope["frozen_git_tree"] != FROZEN_GIT_TREE
            or envelope["manifest_sha256"] != hashlib.sha256(raw).hexdigest()
            or envelope["content_binding"] !=
            "SHA256(context || SHA256(canonical_manifest))"):
        raise ValueError("detached signature does not bind the frozen manifest")
    key = serialization.load_pem_public_key(trusted_public_pem.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("trusted public key must be PEM Ed25519 key")
    if envelope["public_key_sha256"] != _public_fingerprint(key):
        raise ValueError("trusted public key fingerprint mismatch")
    signature = base64.b64decode(envelope["signature_base64"], validate=True)
    try:
        key.verify(signature, _digest(raw))
    except InvalidSignature as exc:
        raise ValueError("cryptographic signature verification failed") from exc
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    signer = command.add_parser("sign")
    signer.add_argument("--manifest", type=Path, required=True)
    signer.add_argument("--private-key", type=Path, required=True)
    signer.add_argument("--out", type=Path, required=True)
    verifier = command.add_parser("verify")
    verifier.add_argument("--manifest", type=Path, required=True)
    verifier.add_argument("--signature", type=Path, required=True)
    verifier.add_argument("--trusted-public-key", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sign":
        sig = sign_detached(args.manifest, args.private_key)
        args.out.write_bytes(canonical_bytes(sig))
        print(json.dumps({
            "signed_manifest_sha256": sig["manifest_sha256"],
            "public_key_sha256": sig["public_key_sha256"],
            "signature_file": str(args.out),
        }, sort_keys=True))
    else:
        print(json.dumps({
            "signature_verified": verify_detached(
                args.manifest, args.signature, args.trusted_public_key
            )
        }, sort_keys=True))


if __name__ == "__main__":
    main()
