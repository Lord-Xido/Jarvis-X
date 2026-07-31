#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/src"

cat fragments/index.html/*.part > index.html
cat fragments/JarvisXMultimodal.ps1/*.part > JarvisXMultimodal.ps1

python3 - <<'PY'
from pathlib import Path

html = Path("index.html").read_bytes()
powershell = Path("JarvisXMultimodal.ps1").read_bytes()

def emit_array(name: str, data: bytes) -> str:
    rows = [f"static const unsigned char {name}[] = {{"]
    for index in range(0, len(data), 16):
        rows.append("  " + ", ".join(f"0x{value:02x}" for value in data[index:index + 16]) + ",")
    rows.extend(["};", f"static const unsigned long {name}_len = {len(data)}UL;"])
    return "\n".join(rows)

Path("embedded_assets.inc").write_text(
    emit_array("APP_HTML", html) + "\n\n" + emit_array("APP_PS1", powershell),
    encoding="ascii",
)
PY

lld-link /lib /def:kernel32.def /machine:x64 /out:kernel32.lib
lld-link /lib /def:user32.def /machine:x64 /out:user32.lib
clang-cl --target=x86_64-pc-windows-msvc \
  /c /O2 /GS- /GR- /EHs-c- /utf-8 /nologo \
  launcher.c /Folauncher.obj
lld-link launcher.obj kernel32.lib user32.lib \
  /entry:wWinMainCRTStartup /subsystem:windows /nodefaultlib /machine:x64 \
  /opt:ref /opt:icf /Brepro \
  /out:"$ROOT/JarvisX-Multimodal-Studio.exe"

sha256sum "$ROOT/JarvisX-Multimodal-Studio.exe"
