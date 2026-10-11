"""Explicit-authority Python bridge to DM3D C++ encrypted byte codec.

The native program uses reversible 3D XOR lifting, lossless zlib and
AES-256-GCM. This control-plane adapter never interprets multimedia content,
passes no secrets via argv, never invokes a shell and is closed by default.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SecureCodecResult:
    operation: str
    committed: bool
    message: str


class DM3DSecureBridge:
    """Publish output only with explicit caller authorization."""

    def __init__(
        self,
        executable: str | os.PathLike[str],
        keyfile: str | os.PathLike[str],
    ) -> None:
        self.executable = Path(executable)
        self.keyfile = Path(keyfile)

    def _run(self, operation: str, *args: str | os.PathLike[str]) -> str:
        completed = subprocess.run(
            [str(self.executable), operation, str(self.keyfile), *(str(a) for a in args)],
            check=False,
            shell=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"DM3D secure {operation} rejected: {completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    def consider(self, input_file: str | os.PathLike[str], mime: str) -> SecureCodecResult:
        """Check the encode/decode inverse without publishing an output."""
        return SecureCodecResult(
            "consider", False, self._run("consider", input_file, mime)
        )

    def seal(
        self,
        input_file: str | os.PathLike[str],
        encrypted_output: str | os.PathLike[str],
        mime: str,
        *,
        authorize_commit: bool = False,
    ) -> SecureCodecResult:
        """Produce a new encrypted container only after an explicit decision."""
        if not authorize_commit:
            raise PermissionError("seal requires authorize_commit=True")
        return SecureCodecResult(
            "seal", True, self._run("seal", input_file, encrypted_output, mime)
        )

    def open(
        self,
        encrypted_input: str | os.PathLike[str],
        recovered_output: str | os.PathLike[str],
        *,
        authorize_commit: bool = False,
    ) -> SecureCodecResult:
        """Publish authenticated decrypted bytes only when authorized."""
        if not authorize_commit:
            raise PermissionError("open requires authorize_commit=True")
        return SecureCodecResult(
            "open", True, self._run("open", encrypted_input, recovered_output)
        )
