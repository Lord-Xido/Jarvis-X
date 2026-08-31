"""Jarvis-X bridge for the 1,000-line Dr Moagi 3D animation codec reference.

The canonical reference source is intentionally stored as five ordered fragments so
its exact 1,000-line form is easy to audit in Git. This module verifies, assembles,
and executes that source without changing its contents.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from typing import Sequence

EXPECTED_LINES = 1000
EXPECTED_SHA256 = "c3d05a11e3fbb91591f538c75bddc15a72dd398e62ae4624a7b1a0828efa620e"
_FRAGMENT_DIR = Path("reference/dr-moagi-3d-animation-autoencoder/fragments")
_RUNTIME_MODULE = "jarvisx._dr_moagi_3d_animation_autoencoder_reference"


def repository_root() -> Path:
    """Return the Jarvis-X source-tree root for a checkout installation."""

    return Path(__file__).resolve().parents[2]


def fragment_directory(root: Path | None = None) -> Path:
    """Resolve the canonical reference fragment directory."""

    base = repository_root() if root is None else Path(root)
    return base / _FRAGMENT_DIR


def reference_fragments(root: Path | None = None) -> list[Path]:
    """Return the five canonical fragments in deterministic order."""

    directory = fragment_directory(root)
    fragments = sorted(directory.glob("*.part"))
    if len(fragments) != 5:
        raise FileNotFoundError(
            f"expected 5 codec fragments in {directory}, found {len(fragments)}"
        )
    return fragments


def reference_source(root: Path | None = None) -> str:
    """Reconstruct the exact canonical 1,000-line Python source."""

    return "".join(path.read_text(encoding="utf-8") for path in reference_fragments(root))


def source_line_count(source: str) -> int:
    """Count physical source lines exactly as stored."""

    return len(source.splitlines())


def source_sha256(source: str) -> str:
    """Return the SHA-256 digest of UTF-8 source bytes."""

    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def verify_reference(root: Path | None = None) -> dict[str, object]:
    """Verify fragment count, physical line count, syntax and source digest."""

    source = reference_source(root)
    lines = source_line_count(source)
    digest = source_sha256(source)
    if lines != EXPECTED_LINES:
        raise RuntimeError(f"codec line-count mismatch: expected {EXPECTED_LINES}, got {lines}")
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"codec SHA-256 mismatch: expected {EXPECTED_SHA256}, got {digest}")
    compile(source, "dr_moagi_3d_animation_autoencoder_1000_lines.py", "exec")
    return {
        "fragments": len(reference_fragments(root)),
        "lines": lines,
        "sha256": digest,
        "syntax": "ok",
    }


def materialize(path: Path, root: Path | None = None) -> Path:
    """Write the verified canonical source to a normal .py file."""

    verify_reference(root)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(reference_source(root), encoding="utf-8", newline="")
    return target


def load_reference_engine(root: Path | None = None) -> types.ModuleType:
    """Load the verified reference engine as a Python module."""

    verify_reference(root)
    if _RUNTIME_MODULE in sys.modules:
        return sys.modules[_RUNTIME_MODULE]
    module = types.ModuleType(_RUNTIME_MODULE)
    module.__file__ = str(fragment_directory(root) / "<assembled>")
    module.__package__ = "jarvisx"
    sys.modules[_RUNTIME_MODULE] = module
    source = reference_source(root)
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(_RUNTIME_MODULE, None)
        raise
    return module


def main(argv: Sequence[str] | None = None) -> int:
    """Verify/materialize the reference or delegate to its native CLI."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--verify-reference":
        status = verify_reference()
        print(
            f"verified fragments={status['fragments']} lines={status['lines']} "
            f"sha256={status['sha256']} syntax={status['syntax']}"
        )
        return 0
    if len(args) == 2 and args[0] == "--materialize":
        target = materialize(Path(args[1]))
        print(f"materialized verified codec to {target}")
        return 0
    engine = load_reference_engine()
    return int(engine.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
