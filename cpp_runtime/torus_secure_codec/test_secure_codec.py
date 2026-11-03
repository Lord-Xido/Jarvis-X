"""Integration and adversarial tests for the self-contained C++ secure torus codec."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(binary, *args, ok=True):
    p = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True)
    if ok and p.returncode:
        raise AssertionError(f"{args}: {p.stderr}")
    if not ok and not p.returncode:
        raise AssertionError(f"unexpected success: {args}")
    return p


def main():
    binary = Path(sys.argv[1]).resolve()
    run(binary, "selftest")
    cases = [b"", b"x", b"\0" * 512, os.urandom(511), os.urandom(512),
             os.urandom(513), os.urandom(4096), os.urandom(600001)]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key, wrong = root / "key.bin", root / "wrong.bin"
        run(binary, "keygen", key)
        run(binary, "keygen", wrong)
        assert len(key.read_bytes()) == 32
        if os.name == "posix":
            assert key.stat().st_mode & 0o077 == 0, "key permissions are not private"
        run(binary, "keygen", key, ok=False)
        for index, data in enumerate(cases):
            plain, cipher, output = (root / f"input{index}", root / f"sealed{index}", root / f"opened{index}")
            plain.write_bytes(data)
            run(binary, "seal", plain, cipher, key)
            assert cipher.read_bytes().startswith(b"JXTOR3D1")
            run(binary, "open", cipher, output, key)
            assert output.read_bytes() == data
            run(binary, "open", cipher, output, key, ok=False)  # no clobber
            assert output.read_bytes() == data
            if index == 5:
                original = cipher.read_bytes()
                for label, pos in [("header", 9), ("nonce", 19), ("body", 32), ("tag", -1)]:
                    altered = bytearray(original)
                    altered[pos] ^= 1
                    corrupt = root / f"corrupt_{label}"
                    recovered = root / f"recovered_{label}"
                    corrupt.write_bytes(altered)
                    run(binary, "open", corrupt, recovered, key, ok=False)
                    assert not recovered.exists(), f"unauthenticated output published: {label}"
                wrong_output = root / "wrong_output"
                run(binary, "open", cipher, wrong_output, wrong, ok=False)
                assert not wrong_output.exists()
                trunc = root / "truncated"
                trunc.write_bytes(original[:-1])
                run(binary, "open", trunc, root / "trunc_output", key, ok=False)
        assert not list(root.glob("*.tmp.*")), "temporary files remained"
    print(f"PASS: {len(cases)} byte-exact round trips; tamper, wrong key, truncation, overwrite, cleanup")


if __name__ == "__main__":
    main()
