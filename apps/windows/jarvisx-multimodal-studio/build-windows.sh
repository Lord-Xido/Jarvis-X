#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/src"

resolve_tool() {
  local requested="$1"
  shift
  if [[ -n "$requested" ]] && command -v "$requested" >/dev/null 2>&1; then
    command -v "$requested"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

CLANG_CL_BIN="$(resolve_tool "${CLANG_CL:-}" clang-cl clang-cl-20 clang-cl-19 clang-cl-18 clang-cl-17 clang-cl-16)" || {
  echo "clang-cl was not found; install LLVM's MSVC-compatible frontend" >&2
  exit 127
}
LLD_LINK_BIN="$(resolve_tool "${LLD_LINK:-}" lld-link lld-link-20 lld-link-19 lld-link-18 lld-link-17 lld-link-16)" || {
  echo "lld-link was not found; install LLVM's COFF linker" >&2
  exit 127
}

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

"$LLD_LINK_BIN" /lib /def:kernel32.def /machine:x64 /out:kernel32.lib
"$LLD_LINK_BIN" /lib /def:user32.def /machine:x64 /out:user32.lib
"$CLANG_CL_BIN" --target=x86_64-pc-windows-msvc \
  /c /O2 /GS- /GR- /EHs-c- /utf-8 /nologo \
  launcher.c /Folauncher.obj
"$LLD_LINK_BIN" launcher.obj kernel32.lib user32.lib \
  /entry:wWinMainCRTStartup /subsystem:windows /nodefaultlib /machine:x64 \
  /opt:ref /opt:icf /Brepro \
  /out:"$ROOT/JarvisX-Multimodal-Studio.exe"

sha256sum "$ROOT/JarvisX-Multimodal-Studio.exe"
