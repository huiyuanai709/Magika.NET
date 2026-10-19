#!/usr/bin/env bash
# Tool-selection regression tests for check-aot-isa.sh.
# Linux/Windows stay on objdump, llvm-objdump, or dumpbin.
# Darwin may use Xcode llvm-objdump (xcrun) or otool only after those miss.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
check="$root/packaging/check-aot-isa.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

ok() {
  echo "ok: $*"
}

sandbox=""
status=0

begin_case() {
  sandbox="$tmp/$1"
  rm -rf "$sandbox"
  mkdir -p "$sandbox/bin" "$sandbox/logs" "$sandbox/work"
  local tool
  for tool in bash mktemp rm awk; do
    ln -s "$(command -v "$tool")" "$sandbox/bin/$tool"
  done
  : > "$sandbox/work/bin"
  : > "$sandbox/logs/objdump"
  : > "$sandbox/logs/llvm-objdump"
  : > "$sandbox/logs/xcrun"
  : > "$sandbox/logs/otool"
  : > "$sandbox/logs/uname"
  FAKE_NEON_LINES=60
  XCRUN_FIND_FAIL=""
  OTOOL_MODE=neon
  FAKE_UNAME_S=Darwin
}

write_stub() {
  local name="$1"
  local body="$2"
  printf '%s\n' '#!/bin/sh' "$body" > "$sandbox/bin/$name"
  chmod +x "$sandbox/bin/$name"
}

add_objdump_stub() {
  local mode="$1"
  if [ "$mode" = "avx2" ]; then
    write_stub objdump "$(cat << 'EOF'
printf '%s\n' "$*" >> "${OBJDUMP_LOG:?}"
for arg in "$@"; do
  if [ "$arg" = "-f" ]; then
    exit 0
  fi
done
i=0
while [ "$i" -lt 120 ]; do
  echo "  vfmadd231ps %ymm0, %ymm1, %ymm2"
  i=$((i + 1))
done
EOF
)"
  else
    write_stub objdump "$(cat << 'EOF'
printf '%s\n' "$*" >> "${OBJDUMP_LOG:?}"
exit 1
EOF
)"
  fi
}

add_llvm_stub() {
  write_stub llvm-objdump "$(cat << 'EOF'
printf '%s\n' "$*" >> "${LLVM_LOG:?}"
i=0
n="${FAKE_NEON_LINES:-60}"
while [ "$i" -lt "$n" ]; do
  echo "100000000: fadd v0.4s, v1.4s, v2.4s"
  i=$((i + 1))
done
EOF
)"
}

add_xcrun_stub() {
  write_stub xcrun "$(cat << 'EOF'
printf '%s\n' "$*" >> "${XCRUN_LOG:?}"
if [ "$1" = "--find" ] && [ "$2" = "llvm-objdump" ]; then
  if [ "${XCRUN_FIND_FAIL:-}" = "1" ]; then
    exit 1
  fi
  echo found
  exit 0
fi
if [ "$1" = "llvm-objdump" ]; then
  i=0
  n="${FAKE_NEON_LINES:-60}"
  while [ "$i" -lt "$n" ]; do
    echo "100000000: fadd v0.4s, v1.4s, v2.4s"
    i=$((i + 1))
  done
  exit 0
fi
exit 1
EOF
)"
}

add_otool_stub() {
  write_stub otool "$(cat << 'EOF'
printf '%s\n' "$*" >> "${OTOOL_LOG:?}"
if [ "${OTOOL_MODE:-neon}" = "avx" ]; then
  i=0
  while [ "$i" -lt 120 ]; do
    echo "  vfmadd231ps %ymm0, %ymm1, %ymm2"
    i=$((i + 1))
  done
  exit 0
fi
i=0
n="${FAKE_NEON_LINES:-60}"
while [ "$i" -lt "$n" ]; do
  printf '%s\n' "0000000100000000	fadd.4s	v0, v1, v2"
  i=$((i + 1))
done
EOF
)"
}

add_uname_stub() {
  write_stub uname "$(cat << 'EOF'
printf '%s\n' "$*" >> "${UNAME_LOG:?}"
if [ "$1" = "-s" ]; then
  printf '%s\n' "${FAKE_UNAME_S:?}"
  exit 0
fi
exit 1
EOF
)"
}

run_case() {
  local isa="$1"
  set +e
  env PATH="$sandbox/bin" \
    OBJDUMP_LOG="$sandbox/logs/objdump" \
    LLVM_LOG="$sandbox/logs/llvm-objdump" \
    XCRUN_LOG="$sandbox/logs/xcrun" \
    OTOOL_LOG="$sandbox/logs/otool" \
    UNAME_LOG="$sandbox/logs/uname" \
    FAKE_NEON_LINES="$FAKE_NEON_LINES" \
    XCRUN_FIND_FAIL="$XCRUN_FIND_FAIL" \
    OTOOL_MODE="$OTOOL_MODE" \
    FAKE_UNAME_S="$FAKE_UNAME_S" \
    "$check" "$sandbox/work/bin" "$isa" \
    > "$sandbox/logs/stdout" 2> "$sandbox/logs/stderr"
  status=$?
  set -e
}

assert_status() {
  if [ "$status" -ne "$1" ]; then
    echo "stdout: $(cat "$sandbox/logs/stdout")" >&2
    echo "stderr: $(cat "$sandbox/logs/stderr")" >&2
    fail "$2 (status $status, expected $1)"
  fi
}

assert_empty() {
  if [ -s "$sandbox/logs/$1" ]; then
    echo "log $1: $(cat "$sandbox/logs/$1")" >&2
    fail "$2"
  fi
}

assert_file_contains() {
  grep -q -F -e "$2" -- "$1" || fail "$3: $(cat "$1")"
}

if ! command -v gcc >/dev/null 2>&1 || ! command -v objdump >/dev/null 2>&1; then
  fail "gcc and objdump are required"
fi

cat > "$tmp/avx2.s" << 'EOF'
.text
.globl f
f:
.rept 120
vfmadd231ps %ymm0, %ymm1, %ymm2
.endr
ret
EOF
cat > "$tmp/avx512.s" << 'EOF'
.text
.globl f
f:
.rept 120
vfmadd231ps %zmm0, %zmm1, %zmm2
.endr
ret
EOF
gcc -c "$tmp/avx2.s" -o "$tmp/avx2.o"
gcc -c "$tmp/avx512.s" -o "$tmp/avx512.o"

"$check" "$tmp/avx2.o" avx2 > "$tmp/avx2.out"
assert_file_contains "$tmp/avx2.out" "isa-check avx2:" "avx2 summary missing"
if "$check" "$tmp/avx2.o" avx512 > "$tmp/avx2-as-512.out" 2> "$tmp/avx2-as-512.err"; then
  fail "avx2 object passed the avx512 check"
fi
assert_file_contains "$tmp/avx2-as-512.err" "does not contain enough zmm" "avx2 object should fail avx512 on zmm count"

"$check" "$tmp/avx512.o" avx512 > "$tmp/avx512.out"
assert_file_contains "$tmp/avx512.out" "isa-check avx512:" "avx512 summary missing"
if "$check" "$tmp/avx512.o" avx2 > "$tmp/avx512-as-2.out" 2> "$tmp/avx512-as-2.err"; then
  fail "avx512 object passed the avx2 check"
fi
assert_file_contains "$tmp/avx512-as-2.err" "contains zmm" "avx512 object should fail avx2"
ok "real objdump still enforces avx2 and avx512"

begin_case objdump-wins
add_objdump_stub avx2
add_uname_stub
add_xcrun_stub
add_otool_stub
run_case avx2
assert_status 0 "objdump on PATH should satisfy avx2 without Darwin tools"
assert_empty uname "Darwin detection ran before objdump"
assert_empty xcrun "xcrun ran even though objdump works"
assert_empty otool "otool ran even though objdump works"
ok "objdump stays ahead of xcrun and otool"

begin_case llvm-on-path
add_objdump_stub fail
add_llvm_stub
add_uname_stub
add_xcrun_stub
add_otool_stub
run_case apple-m1
assert_status 0 "PATH llvm-objdump should satisfy apple-m1"
assert_empty uname "Darwin detection ran before PATH llvm-objdump"
assert_empty xcrun "xcrun ran even though llvm-objdump is on PATH"
assert_empty otool "otool ran even though llvm-objdump is on PATH"
assert_file_contains "$sandbox/logs/llvm-objdump" "-d" "PATH llvm-objdump was not asked to disassemble"
ok "PATH llvm-objdump stays ahead of xcrun and otool"

begin_case xcrun
add_uname_stub
add_xcrun_stub
add_otool_stub
FAKE_NEON_LINES=50
run_case apple-m1
assert_status 0 "50 AdvSimd lines from xcrun llvm-objdump should pass"
assert_empty otool "otool ran when xcrun llvm-objdump was available"
assert_file_contains "$sandbox/logs/xcrun" "llvm-objdump -d" "xcrun was not asked to disassemble"
assert_file_contains "$sandbox/logs/stdout" "neon=50" "neon count was not 50"
FAKE_NEON_LINES=49
: > "$sandbox/logs/stdout"
: > "$sandbox/logs/stderr"
: > "$sandbox/logs/xcrun"
run_case apple-m1
assert_status 1 "49 AdvSimd lines should fail apple-m1"
assert_file_contains "$sandbox/logs/stderr" "does not contain enough AdvSimd" "short disassembly used the wrong error"
ok "Darwin uses xcrun llvm-objdump and keeps the AdvSimd threshold"

begin_case otool
add_uname_stub
add_xcrun_stub
add_otool_stub
XCRUN_FIND_FAIL=1
FAKE_NEON_LINES=50
run_case apple-m1
assert_status 0 "otool -tV should satisfy apple-m1 when xcrun cannot find llvm-objdump"
assert_file_contains "$sandbox/logs/otool" "-tV" "otool was not asked for text disassembly"
assert_file_contains "$sandbox/logs/stdout" "neon=50" "otool syntax was not counted"
ok "Darwin falls back to otool"

begin_case linux-ignores-otool
ln -s "$(command -v uname)" "$sandbox/bin/uname"
add_otool_stub
OTOOL_MODE=avx
run_case avx2
assert_status 1 "Linux must not accept otool as an AVX disassembler"
assert_empty otool "otool ran on Linux"
assert_file_contains "$sandbox/logs/stderr" "need objdump, llvm-objdump, or dumpbin to check" "Linux error text changed"
: > "$sandbox/logs/stdout"
: > "$sandbox/logs/stderr"
: > "$sandbox/logs/otool"
run_case apple-m1
assert_status 1 "Linux must not use otool for apple-m1"
assert_empty otool "otool ran on Linux for apple-m1"
assert_file_contains "$sandbox/logs/stderr" "need objdump, llvm-objdump, or dumpbin to check" "Linux apple-m1 error text changed"
ok "Linux still requires objdump, llvm-objdump, or dumpbin"

begin_case darwin-missing
add_uname_stub
run_case apple-m1
assert_status 1 "Darwin without a disassembler should fail"
assert_file_contains "$sandbox/logs/stderr" "need Xcode llvm-objdump (xcrun llvm-objdump) or otool to check" "Darwin error text missing"
ok "Darwin reports xcrun llvm-objdump or otool when both are missing"

release="$root/.github/workflows/release.yml"
assert_file_contains "$release" "xcrun --find llvm-objdump" "release workflow does not look up Xcode llvm-objdump"
assert_file_contains "$release" "exec xcrun llvm-objdump" "release workflow does not expose llvm-objdump through xcrun"
if grep -q "sku: osx-arm64" "$release" && grep -q "isa: apple-m1" "$release"; then
  ok "osx-arm64 sku and apple-m1 isa are unchanged"
else
  fail "osx-arm64 matrix entry changed"
fi

ok "all isa-check tool selection cases passed"
