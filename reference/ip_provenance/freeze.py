"""Reproducible, byte-exact Git tree IP baseline witness (no modifications).

The SHA-1 Git object name identifies the frozen historical tree; this tool
independently hashes every tracked Git blob with SHA-256 and commits it to a
domain-separated SHA-256 Merkle root. It reads object bytes *at the frozen
commit*, not mutable working-tree files.

This is integrity/provenance evidence, NOT a digital author signature, a
trusted independent timestamp, an assignment of ownership, or patent filing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SCHEMA = "JarvisX/IPFreeze/1"
REPOSITORY = "Lord-Xido/Jarvis-X"
FROZEN_COMMIT = "b511300d565cc13865e3c2026050538bf9e2b149"
FROZEN_GIT_TREE = "302ae87d36bd00d57fddd350c7ecf44bc4f92821"
LEAF_PREFIX = b"JARVISX-IPFREEZE/v1/leaf\x00"
NODE_PREFIX = b"JARVISX-IPFREEZE/v1/node\x00"
ROOT_FILE = Path(__file__).resolve().parents[2] / "docs" / "ip" / "2026-10-29-root.sha256"
MANIFEST_HASH_FILE = ROOT_FILE.with_name("2026-10-29-manifest.sha256")


def _git(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("Git object read failed; complete history is required: "
                           + " ".join(args[:3]))
    return completed.stdout


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _leaf(path: bytes, mode: bytes, payload_digest: bytes, length: int) -> bytes:
    return _sha256(
        LEAF_PREFIX + len(path).to_bytes(4, "big") + path +
        len(mode).to_bytes(1, "big") + mode +
        length.to_bytes(8, "big") + payload_digest
    )


def merkle(leaves: list[bytes]) -> str:
    if not leaves:
        raise ValueError("empty baseline cannot be attested")
    if any(type(leaf) is not bytes or len(leaf) != 32 for leaf in leaves):
        raise ValueError("invalid leaf digest")
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])  # Explicit duplicate-last rule.
        level = [
            _sha256(NODE_PREFIX + level[index] + level[index + 1])
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def capture() -> dict:
    tree = _git("rev-parse", FROZEN_COMMIT + "^{tree}").strip().decode("ascii")
    if tree != FROZEN_GIT_TREE:
        raise RuntimeError("frozen baseline Git tree does not match original commit")
    raw_listing = _git("ls-tree", "-rz", "--full-tree", FROZEN_COMMIT)
    entries: list[dict] = []
    leaves: list[bytes] = []
    total_bytes = 0
    records: list[tuple[bytes, bytes, bytes, bytes]] = []
    for row in raw_listing.split(b"\x00"):
        if not row:
            continue
        try:
            meta, path = row.split(b"\t", 1)
            mode, kind, oid = meta.split(b" ")
        except ValueError as exc:
            raise RuntimeError("invalid Git tree listing") from exc
        if kind != b"blob" or mode not in (b"100644", b"100755", b"120000") or len(oid) != 40 or not path:
            raise RuntimeError("unsupported frozen Git tree object")
        records.append((path, mode, kind, oid))
    records.sort(key=lambda item: item[0])  # Unambiguous raw-byte ordering.
    previous: bytes | None = None
    for path, mode, kind, oid in records:
        if previous is not None and path == previous:
            raise RuntimeError("duplicate frozen Git tree path")
        previous = path
        # Resolve object by immutable OID. No symlink is followed.
        data = _git("cat-file", "blob", oid.decode("ascii"))
        total_bytes += len(data)
        # The JSON path is supplementary; the exact path bytes determine hash.
        try:
            path_text = path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("non-UTF8 tree paths require a new manifest schema") from exc
        content_sha = _sha256(data)
        leaves.append(_leaf(path, mode, content_sha, len(data)))
        entries.append({
            "path": path_text,
            "git_mode": mode.decode("ascii"),
            "git_blob_sha1": oid.decode("ascii"),
            "size_bytes": len(data),
            "sha256": content_sha.hex(),
        })
    root = merkle(leaves)
    report = {
        "schema": SCHEMA,
        "repository": REPOSITORY,
        "frozen_git_commit": FROZEN_COMMIT,
        "frozen_git_tree": FROZEN_GIT_TREE,
        "file_count": len(entries),
        "total_blob_bytes": total_bytes,
        "sha256_merkle_root": root,
        "merkle_leaf_definition": (
            "SHA256(LEAF_PREFIX || u32be(path_length) || path_bytes || "
            "u8(mode_length) || mode_ascii || u64be(blob_length) || "
            "SHA256(git_blob_bytes))"
        ),
        "merkle_node_definition": (
            "SHA256(NODE_PREFIX || left32 || right32), sorted raw Git path "
            "bytes ascending; duplicate final node if level count odd"
        ),
        "coverage": (
            "All tracked Git blobs at frozen commit, including executable bits "
            "and symlink blob contents; no build artifacts, unreachable Git "
            "objects, external dependencies or untracked/private files"
        ),
        "cryptographic_limits": (
            "Content integrity only. GitHub's verified commit signature does "
            "not prove legal inventorship; this is not a user-held private-key "
            "signature, RFC3161 timestamp or patent registration."
        ),
        "production_gate": "BLOCKED",
        "files": entries,
    }
    if not entries:
        raise RuntimeError("no tracked Git blobs found")
    return report


def canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def verify_against_anchor(result: dict, root_path: Path = ROOT_FILE) -> str:
    if not root_path.is_file():
        raise RuntimeError("fixed SHA-256 root anchor missing; cannot certify baseline")
    expected = root_path.read_text("ascii").strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise RuntimeError("invalid frozen SHA-256 root anchor")
    if result["sha256_merkle_root"] != expected:
        raise RuntimeError("baseline SHA-256 Merkle root differs from fixed anchor")
    return expected


def verify_manifest_digest(raw: bytes, path: Path = MANIFEST_HASH_FILE) -> str:
    if not path.is_file():
        raise RuntimeError("fixed canonical manifest digest anchor missing")
    expected = path.read_text("ascii").strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise RuntimeError("invalid frozen canonical manifest digest anchor")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("manifest SHA-256 differs from fixed canonical inventory anchor")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path, help="compare exact canonical manifest bytes")
    parser.add_argument("--require-anchor", action="store_true")
    args = parser.parse_args()
    result = capture()
    canonical = canonical_bytes(result)
    if args.require_anchor:
        verify_against_anchor(result)
        verify_manifest_digest(canonical)
    if args.verify is not None and args.verify.read_bytes() != canonical:
        raise RuntimeError("provided IP manifest differs from frozen Git tree")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(canonical)
    print(json.dumps({
        "schema": SCHEMA, "commit": FROZEN_COMMIT,
        "tree": FROZEN_GIT_TREE, "files": result["file_count"],
        "bytes": result["total_blob_bytes"],
        "sha256_merkle_root": result["sha256_merkle_root"],
        "manifest_sha256": _sha256(canonical).hex(),
        "anchor_verified": args.require_anchor,
        "output": str(args.out) if args.out else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
