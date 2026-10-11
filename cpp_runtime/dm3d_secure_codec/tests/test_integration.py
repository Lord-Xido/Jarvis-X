#!/usr/bin/env python3
"""Integration: exact byte round-trip, suspension and fail-closed authenticated IO."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

EXE = Path(sys.argv[1]).resolve()


def run(*args, success=True):
    result = subprocess.run([str(EXE), *map(str, args)], capture_output=True, text=True)
    if success and result.returncode:
        raise AssertionError(f"{args}: {result.stderr}")
    if not success and result.returncode == 0:
        raise AssertionError(f"unexpected success: {args}")
    return result


def test():
    with tempfile.TemporaryDirectory(prefix="dm3d-secure-") as d:
        root = Path(d)
        key, wrong = root/"key", root/"wrong"
        run("keygen", key)
        run("keygen", wrong)
        assert len(key.read_bytes()) == 32
        assert key.stat().st_mode & 0o777 == 0o600
        run("keygen", key, success=False)

        wavpath = root/"voice.wav"
        with wave.open(str(wavpath), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(8000)
            out.writeframes(bytes((i*19) % 256 for i in range(16000)))
        samples = [
            ("empty", b"", "application/octet-stream"),
            ("one", b"\\x01", "application/octet-stream"),
            ("utf8", ("你好 مرحبا 😀 Dr Moagi\\n" * 40).encode(), "text/plain"),
            ("image", b"P6\\n16 16\\n255\\n" + os.urandom(768), "image/x-portable-pixmap"),
            ("wav", wavpath.read_bytes(), "audio/wav"),
            ("mp4-bytes", b"\\0\\0\\0\\x18ftypisom" + os.urandom(5409), "video/mp4"),
            ("jpeg-bytes", b"\\xff\\xd8\\xff" + os.urandom(3000), "image/jpeg"),
            ("zeroes", bytes(65536), "application/octet-stream"),
            ("512", os.urandom(512), "application/octet-stream"),
            ("513", os.urandom(513), "application/octet-stream"),
            ("511", os.urandom(511), "application/octet-stream"),
            ("large", os.urandom(1024*1024 + 3), "application/octet-stream"),
        ]
        for name, data, mime in samples:
            raw, sealed, restored = (root/(name+x) for x in (".raw", ".dm3d", ".restored"))
            raw.write_bytes(data)
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            assert "uncommitted" in run("consider", key, raw, mime).stdout
            assert not sealed.exists()
            run("seal", key, raw, sealed, mime)
            run("open", key, sealed, restored)
            assert restored.read_bytes() == data
            assert hashlib.sha256(restored.read_bytes()).hexdigest() == digest
            run("open", key, sealed, restored, success=False)  # no overwrite

        sample = (root/"utf8.dm3d").read_bytes()
        for name, offset in (
            ("ciphertext", 47), ("tag", -1), ("header", 32), ("metadata", 44)
        ):
            changed = bytearray(sample)
            changed[offset] ^= 1
            broken, dest = root/(name+".dm3d"), root/(name+".out")
            broken.write_bytes(changed)
            run("open", key, broken, dest, success=False)
            assert not dest.exists()
        for name, changed in (
            ("truncated", sample[:-1]), ("trailing", sample + b"x")
        ):
            broken, dest = root/(name+".dm3d"), root/(name+".out")
            broken.write_bytes(changed)
            run("open", key, broken, dest, success=False)
            assert not dest.exists()
        # Later-record failure must not publish earlier authenticated fragments.
        corrupted = bytearray(sample)
        cursor = 43 + corrupted[42]
        tags = []
        for _ in range(12):
            if cursor+5 > len(corrupted):
                break
            n = int.from_bytes(corrupted[cursor+3:cursor+5], "big")
            tag = cursor+5+n
            if tag+16 > len(corrupted):
                break
            tags.append(tag)
            cursor = tag+16
        assert len(tags)>1
        corrupted[tags[-1]+8] ^= 1
        target, dest = root/"late.dm3d", root/"late.out"
        target.write_bytes(corrupted)
        run("open", key, target, dest, success=False)
        assert not dest.exists()
        assert not list(root.glob("late.out.tmp.*"))
        run("open", wrong, root/"utf8.dm3d", root/"wrongkey.out", success=False)
        assert not (root/"wrongkey.out").exists()
        insecure = root/"insecure-key"
        insecure.write_bytes(key.read_bytes())
        insecure.chmod(0o644)
        run("consider", insecure, root/"utf8.raw", "text/plain", success=False)
        print("PASS: 12 exact binary round-trips + authorization/tamper tests")


if __name__ == "__main__":
    test()
