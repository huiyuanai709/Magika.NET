#!/usr/bin/env bash
# Run a published Native AOT binary when this CPU can execute it.
# x64 builds are also disassembled for ymm (avx2) or zmm (avx512).
# Usage: smoke-aot.sh <binary> <avx2|avx512|> <fixture>
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: smoke-aot.sh <binary> <avx2|avx512|> <fixture>" >&2
  exit 2
fi

binary="$1"
isa="$2"
fixture="$3"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "$isa" ]; then
  "$root/packaging/check-aot-isa.sh" "$binary" "$isa"
fi

host_has_isa() {
  case "$1" in
    "") return 0 ;;
    avx2|avx512) ;;
    *)
      echo "unknown isa: $1" >&2
      return 2
      ;;
  esac
  if [ -r /proc/cpuinfo ]; then
    case "$1" in
      avx2) grep -qw avx2 /proc/cpuinfo ;;
      avx512) grep -qw avx512f /proc/cpuinfo ;;
    esac
    return
  fi
  local ps=""
  if command -v pwsh >/dev/null 2>&1; then
    ps="pwsh"
  elif command -v powershell.exe >/dev/null 2>&1; then
    ps="powershell.exe"
  elif command -v powershell >/dev/null 2>&1; then
    ps="powershell"
  else
    echo "cannot detect CPU features; skipping execution of ${binary}" >&2
    return 1
  fi
  local type="Avx2"
  if [ "$1" = "avx512" ]; then
    type="Avx512F"
  fi
  set +e
  "$ps" -NoProfile -Command "try { if ([System.Runtime.Intrinsics.X86.${type}]::IsSupported) { exit 0 } else { exit 1 } } catch { exit 2 }"
  local status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    return 0
  fi
  if [ "$status" -eq 2 ]; then
    echo "CPU feature probe failed; skipping execution of ${binary}" >&2
  fi
  return 1
}

if ! host_has_isa "$isa"; then
  echo "host cannot run isa '${isa:-any}'; disassembly check already ran when required" >&2
  exit 0
fi

"$binary" --version
"$binary" --help >/dev/null
out="$("$binary" "$fixture")"
echo "$out"
label="${out#*: }"
label="${label%% *}"
if [ "$label" != "pdf" ]; then
  echo "binary did not classify the PDF fixture as pdf: ${out}" >&2
  exit 1
fi
