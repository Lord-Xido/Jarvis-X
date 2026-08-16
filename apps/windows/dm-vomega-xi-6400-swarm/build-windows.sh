#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CC="${CC:-clang}"
LLD="${LLD:-lld-link}"
DLLTOOL="${DLLTOOL:-llvm-dlltool}"
SRC=dm_vomega_xi_6400gib_swarm_isa.c
OBJ=dm_vomega_xi_6400gib_swarm_isa.obj
OUT=DM_vOmegaXi_6400GB_3D_VRAM_Swarm_ISA.exe

cat > kernel32.def <<'DEF'
LIBRARY KERNEL32.dll
EXPORTS
GetStdHandle
WriteFile
Sleep
ExitProcess
DEF

"$DLLTOOL" -m i386:x86-64 -d kernel32.def -l kernel32.lib
"$CC" --target=x86_64-pc-windows-msvc -O2 -ffreestanding -fno-builtin -fno-stack-protector -c "$SRC" -o "$OBJ"
"$LLD" /Brepro /entry:start /subsystem:console /nodefaultlib /dynamicbase /nxcompat /out:"$OUT" "$OBJ" kernel32.lib
sha256sum "$OUT" > SHA256SUMS.txt
