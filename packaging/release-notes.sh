#!/usr/bin/env bash
# Write GitHub Release notes for one tag to stdout.
# Usage: release-notes.sh v0.1.0
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: release-notes.sh <tag>" >&2
  exit 2
fi

tag="$1"
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

prerelease_block=""
case "$tag" in
  *-*)
    prerelease_block="这是预发布版本，不会标成 Latest。

"
    ;;
esac

notes=$(cat <<'EOF'
__PRERELEASE_BLOCK__Magika.NET `__TAG__`。Native AOT 命令行，模型 standard_v3_3 已嵌进二进制。许可是 Apache-2.0。每个 zip 里有 `RUN.txt`。

| 文件 | 说明 |
| --- | --- |
| `magika-__TAG__-linux-x64-avx2.zip` | Linux x64，AVX2。在 ubuntu-latest 上构建，需要兼容的 glibc 和 AVX2。解压后运行 `./magika <文件>`。 |
| `magika-__TAG__-linux-x64-avx512.zip` | Linux x64，AVX-512。需要 AVX-512，只有 AVX2 的机器不能运行。 |
| `magika-__TAG__-win-x64-avx2.zip` | Windows x64，AVX2。解压后运行 `magika.exe <文件>`。 |
| `magika-__TAG__-win-x64-avx512.zip` | Windows x64，AVX-512。需要 AVX-512。 |
| `magika-__TAG__-osx-arm64.zip` | macOS Apple Silicon（M 系列，arm64，apple-m1 / AdvSimd）。在 macos-latest 上构建。解压后运行 `./magika <文件>`。 |
| `SHA256SUMS.txt` | 上面五个 zip 的 SHA-256。 |

文本输出是「路径: 标签 分数 MIME 分组」。`--help` 打印其余参数。构建时已经用 PDF 样本跑过分类。
EOF
)

notes="${notes//__TAG__/$tag}"
notes="${notes//__PRERELEASE_BLOCK__/$prerelease_block}"
printf '%s\n' "$notes"
