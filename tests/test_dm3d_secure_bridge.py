"""Closed-by-default authority and subprocess isolation for DM3D-SE."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvisx.dm3d_secure_bridge import DM3DSecureBridge


def test_closed_gate_never_invokes_native_program(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native program must not execute")

    monkeypatch.setattr("jarvisx.dm3d_secure_bridge.subprocess.run", forbidden)
    bridge = DM3DSecureBridge("codec", "key")
    with pytest.raises(PermissionError, match="authorize_commit"):
        bridge.seal("in.wav", "out.dm3d", "audio/wav")
    with pytest.raises(PermissionError, match="authorize_commit"):
        bridge.open("encrypted.dm3d", "decoded.wav")


def test_consider_remains_uncommitted_and_avoids_shell(monkeypatch):
    invocations = []

    def stub(argv, **kwargs):
        invocations.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0, stdout="CONSIDERED (uncommitted)", stderr=""
        )

    monkeypatch.setattr("jarvisx.dm3d_secure_bridge.subprocess.run", stub)
    bridge = DM3DSecureBridge("/bin/codec", "/keys/master")
    result = bridge.consider("/data/movie.mp4", "video/mp4")
    assert result.operation == "consider" and not result.committed
    assert invocations[0][0] == [
        "/bin/codec", "consider", "/keys/master", "/data/movie.mp4", "video/mp4"
    ]
    assert invocations[0][1]["shell"] is False


def test_explicit_commit_and_failure_propagation(monkeypatch):
    calls = []

    def accepted(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="OK", stderr="")

    monkeypatch.setattr("jarvisx.dm3d_secure_bridge.subprocess.run", accepted)
    bridge = DM3DSecureBridge("codec", "master")
    assert bridge.seal("in", "enc", "application/octet-stream", authorize_commit=True).committed
    assert calls[-1][1] == "seal"
    assert bridge.open("enc", "out", authorize_commit=True).committed
    assert calls[-1][1] == "open"

    def rejected(argv, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="", stderr="AUTHENTICATION FAILURE"
        )

    monkeypatch.setattr("jarvisx.dm3d_secure_bridge.subprocess.run", rejected)
    with pytest.raises(RuntimeError, match="AUTHENTICATION FAILURE"):
        bridge.open("corrupt.dm3d", "out", authorize_commit=True)
