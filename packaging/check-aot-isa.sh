#!/usr/bin/env bash
# Confirm a Native AOT binary actually contains the requested codegen.
# Usage: check-aot-isa.sh <binary> <avx2|avx512|apple-m1>
# avx2 requires ymm and vfmadd and no zmm. avx512 requires zmm.
# apple-m1 requires AdvSimd float ops (.4s / fmla / fadd).
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: check-aot-isa.sh <binary> <avx2|avx512|apple-m1>" >&2
  exit 2
fi

binary="$1"
isa="$2"
if [ ! -f "$binary" ]; then
  echo "missing binary: ${binary}" >&2
  exit 1
fi

case "$isa" in
  avx2|avx512|apple-m1) ;;
  *)
    echo "isa must be avx2, avx512, or apple-m1" >&2
    exit 2
    ;;
esac

to_bash_path() {
  local path="$1"
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -u "$path"
    return
  fi
  printf '%s\n' "$path"
}

find_dumpbin() {
  local vswhere=""
  local candidate
  for candidate in \
    "/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" \
    "/mnt/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
  do
    if [ -f "$candidate" ]; then
      vswhere="$candidate"
      break
    fi
  done
  if [ -z "$vswhere" ]; then
    return 1
  fi
  local install
  install="$("$vswhere" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | tr -d '\r')"
  if [ -z "$install" ]; then
    return 1
  fi
  install="$(to_bash_path "$install")"
  local latest dump
  latest="$(find "$install/VC/Tools/MSVC" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1)"
  dump="${latest}/bin/Hostx64/x64/dumpbin.exe"
  if [ -f "$dump" ]; then
    printf '%s\n' "$dump"
    return 0
  fi
  return 1
}

disassemble() {
  if command -v objdump >/dev/null 2>&1 && objdump -f "$binary" >/dev/null 2>&1; then
    objdump -d --no-show-raw-insn "$binary"
    return
  fi
  if command -v llvm-objdump >/dev/null 2>&1; then
    llvm-objdump -d "$binary"
    return
  fi
  local dumpbin
  if dumpbin="$(find_dumpbin)"; then
    "$dumpbin" /DISASM /NOLOGO "$binary"
    return
  fi
  echo "need objdump, llvm-objdump, or dumpbin to check ${binary}" >&2
  return 1
}

tmp="$(mktemp)"
cleanup() {
  rm -f "$tmp"
}
trap cleanup EXIT

# Stream the disassembly. A full listing is large; only the counts are kept.
disassemble | awk '
  BEGIN { IGNORECASE = 1 }
  {
    line = tolower($0)
    if (line ~ /ymm/) ymm++
    if (line ~ /zmm/) zmm++
    if (line ~ /vfmadd/) fma++
    if (line ~ /\.4s/ || line ~ /fmla/ || line ~ /fadd/) neon++
  }
  END { printf "%d %d %d %d\n", ymm + 0, zmm + 0, fma + 0, neon + 0 }
' > "$tmp"

read -r ymm zmm fma neon < "$tmp"
echo "isa-check ${isa}: ymm=${ymm} zmm=${zmm} vfmadd=${fma} neon=${neon} binary=${binary}"

case "$isa" in
  avx2)
    if [ "$ymm" -lt 100 ] || [ "$fma" -lt 1 ] || [ "$zmm" -ne 0 ]; then
      echo "avx2 Native AOT binary is missing ymm/vfmadd or contains zmm (SSE-only or wrong ISA)" >&2
      exit 1
    fi
    ;;
  avx512)
    if [ "$zmm" -lt 100 ]; then
      echo "avx512 Native AOT binary does not contain enough zmm instructions" >&2
      exit 1
    fi
    ;;
  apple-m1)
    if [ "$neon" -lt 50 ]; then
      echo "apple-m1 Native AOT binary does not contain enough AdvSimd float ops" >&2
      exit 1
    fi
    ;;
esac
