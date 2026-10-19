#!/usr/bin/env bash
# Stage a Native AOT publish directory into a release zip.
# Usage: stage-release.sh <rid> <tag> <publish-dir> <zip-path> [isa]
# isa is avx2 or avx512 for linux-x64 and win-x64, and apple-m1 for osx-arm64.
# The macOS zip name stays osx-arm64 (one Apple Silicon SKU).
# The zip contains magika (or magika.exe), RUN.txt, LICENSE, THIRD-PARTY-NOTICES.md, and VERSION.txt.
set -euo pipefail

if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
  echo "usage: stage-release.sh <rid> <tag> <publish-dir> <zip-path> [avx2|avx512|apple-m1]" >&2
  exit 2
fi

rid="$1"
tag="$2"
publish_dir="$3"
zip_path="$4"
isa="${5:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$rid" in
  linux-x64|osx-arm64) published="Magika.Cli"; binary="magika"; expect_exec=1 ;;
  win-x64) published="Magika.Cli.exe"; binary="magika.exe"; expect_exec=0 ;;
  *)
    echo "unsupported rid: ${rid}" >&2
    exit 2
    ;;
esac

case "$rid" in
  linux-x64|win-x64)
    case "$isa" in
      avx2|avx512) sku="${rid}-${isa}" ;;
      *)
        echo "x64 releases require isa avx2 or avx512" >&2
        exit 2
        ;;
    esac
    ;;
  osx-arm64)
    case "$isa" in
      ""|apple-m1) ;;
      *)
        echo "osx-arm64 isa must be apple-m1" >&2
        exit 2
        ;;
    esac
    sku="$rid"
    isa="apple-m1"
    ;;
esac

case "$tag" in
  v*.*.*) ;;
  *)
    echo "tag must look like v0.1.0" >&2
    exit 2
    ;;
esac
case "$tag" in
  *[!A-Za-z0-9._-]*)
    echo "unsafe tag: ${tag}" >&2
    exit 2
    ;;
esac

if [ ! -f "$publish_dir/$published" ]; then
  echo "missing published binary: ${publish_dir}/${published}" >&2
  exit 1
fi

stage="$(mktemp -d)"
cleanup() {
  rm -rf "$stage"
}
trap cleanup EXIT

cp "$publish_dir/$published" "$stage/$binary"
if [ "$expect_exec" -eq 1 ]; then
  chmod +x "$stage/$binary"
fi

cp "$root/LICENSE" "$root/THIRD-PARTY-NOTICES.md" "$root/packaging/RUN.txt" "$stage/"
printf 'tag=%s\nrid=%s\nisa=%s\nbinary=%s\nlicense=Apache-2.0\nmodel=standard_v3_3\n' "$tag" "$rid" "$isa" "$binary" > "$stage/VERSION.txt"

mkdir -p -- "$(dirname -- "$zip_path")"
case "$zip_path" in
  /*) ;;
  *) zip_path="${PWD}/${zip_path}" ;;
esac

if command -v python3 >/dev/null 2>&1; then
  py=python3
elif command -v python >/dev/null 2>&1; then
  py=python
else
  echo "python3 is required to build the release zip." >&2
  exit 1
fi

STAGE_DIR="$stage" \
  ZIP_PATH="$zip_path" \
  ZIP_PREFIX="magika-${tag}-${sku}" \
  BINARY_NAME="$binary" \
  EXPECT_EXEC="$expect_exec" \
  "$py" - <<'PY'
import os
import sys
import zipfile

stage = os.environ["STAGE_DIR"]
dest = os.environ["ZIP_PATH"]
prefix = os.environ["ZIP_PREFIX"]
binary = os.environ["BINARY_NAME"]
expect_exec = os.environ["EXPECT_EXEC"] == "1"
exec_attr = 0o100755 << 16
file_attr = 0o100644 << 16

os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
if os.path.exists(dest):
    os.remove(dest)

with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(os.listdir(stage)):
        path = os.path.join(stage, name)
        if not os.path.isfile(path):
            continue
        arcname = prefix + "/" + name
        archive.write(path, arcname)
        info = archive.getinfo(arcname)
        info.create_system = 3
        if expect_exec and name == binary:
            info.external_attr = exec_attr
        else:
            info.external_attr = file_attr

    names = set(archive.namelist())
    required = [
        prefix + "/RUN.txt",
        prefix + "/LICENSE",
        prefix + "/THIRD-PARTY-NOTICES.md",
        prefix + "/VERSION.txt",
        prefix + "/" + binary,
    ]
    missing = [name for name in required if name not in names]
    if missing:
        sys.exit("zip missing " + ", ".join(missing))

    symbol_suffixes = (".pdb", ".dbg", ".dwo", ".debug", ".sym", ".dsym")
    symbols = sorted(name for name in names if name.lower().endswith(symbol_suffixes))
    if symbols:
        sys.exit("zip contains debug symbols: " + ", ".join(symbols))

    if expect_exec:
        mode = archive.getinfo(prefix + "/" + binary).external_attr >> 16
        if mode & 0o111 == 0:
            sys.exit(binary + " is not marked executable in the zip")
PY

echo "Wrote ${zip_path}"
