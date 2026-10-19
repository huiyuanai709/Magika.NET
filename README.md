# Magika.NET

.NET 11 上的 [Google Magika](https://github.com/google/magika) 移植。用内置的 `standard_v3_3` 模型判断文件内容类型（PDF、PNG、ELF、ZIP、纯文本等），推理在纯托管代码里完成，不依赖 ONNX Runtime，也没有本机二进制。

## 库

```csharp
using Magika;

MagikaSession session = MagikaSession.Create();
MagikaResult result = session.Identify(File.ReadAllBytes(path));

Console.WriteLine(result.ContentType.Label);      // pdf
Console.WriteLine(result.ContentType.MimeType);   // application/pdf
Console.WriteLine(result.ContentType.Group);      // document
Console.WriteLine(result.Score);                  // 模型给出的最大概率
```

`Identify` 接受 `ReadOnlySpan<byte>`。`Identify(Stream)` 会恢复流的位置；可查找的流只读开头和结尾各一块。`IdentifyPath` 会把目录标成 `directory`。符号链接默认跟随目标，`FollowSymlinks = false` 时标成 `symlink`。

```csharp
MagikaSession session = MagikaSession.Create(new MagikaOptions
{
    PredictionMode = PredictionMode.MediumConfidence,
    FollowSymlinks = false,
});
```

`PredictionMode` 与上游一致：

| 模式 | 行为 |
| --- | --- |
| `HighConfidence`（默认） | 分数达到该标签自己的高置信阈值才保留 |
| `MediumConfidence` | 分数达到模型的中等阈值（`standard_v3_3` 为 0.5）才保留 |
| `BestGuess` | 始终保留模型的最佳标签（仍会走 overwrite map） |

高置信失败、且最终标签被当成文本时，回落为 `txt`，否则为 `unknown`，`OverwriteReason` 为 `LowConfidence`。`randombytes` 会改写成 `unknown`，`randomtxt` 会改写成 `txt`。分数仍是改写前的最大概率。

过短的输入不跑模型：空文件是 `empty`；能通过 UTF-8 检查的短内容是 `txt`，否则是 `unknown`。

## 命令行

程序集名称是 `Magika.Cli`（CLR 的程序集名不区分大小写，不能再叫 `Magika`）。作为 dotnet tool 安装时，命令名是 `magika`。

```bash
dotnet run --project src/Magika.Cli -c Release -- --json sample.pdf
dotnet run --project src/Magika.Cli -c Release -- --prediction-mode medium --no-dereference ./files
```

文本输出：`路径: 标签 分数 MIME 分组`。`--version` 打印库版本和模型名。找不到文件时退出码为 1，参数错误为 2。

## 模型

默认模型是上游当前的 [`standard_v3_3`](https://github.com/google/magika/tree/main/assets/models/standard_v3_3)（214 类）。`model.onnx` 与 `config.min.json` 以嵌入资源的形式放进 `Magika` 程序集，来源和 Apache-2.0 / Google 版权说明在 `assets/models/standard_v3_3/ORIGIN.txt`。

特征与 Rust 实现一致：从文件开头、结尾各取最多 4096 字节，去掉开头（以及结尾）的空白后，拼成 1024 + 1024 个整数 token，不足处用 padding token `256`。`standard_v3_3` 的 `mid_size` 为 0，不使用中间窗口。

## ONNX 如何执行

图由本仓库里的托管解释器执行（`src/Magika/Onnx`）。做法对齐 [SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR)：纯 C# 逐算子解释，GEMM / 卷积用自己写的 AVX-512、AVX2/FMA 内核，其它 CPU（包括 Apple Silicon）走 `System.Numerics.Vector<float>`。没有 ONNX Runtime、Paddle Inference，也没有其它本机推理库。

没有把 SimdPaddleOCR 当成包或子模块引用。它的解释器面向 OCR 图，缺少本模型要用的 `Equal`、`Cast`、`Shape`、`Expand`、`Reciprocal`、`GlobalMaxPool`。这里只实现了 `standard_v3_3` 图里出现的算子。

加载时把「与 `0..256` 比较再 one-hot 乘嵌入矩阵」融合成一次按行读取（`Embedding`）。这一步在 float32 下与原来的 MatMul 相同，也是测试锁定的行为。

卷积是图里最重的算子：权重 `(512, 256, 5, 1)`，约 3.3 亿次乘加。在一台 4 vCPU 的 Xeon 上，预热之后单文件大约 20 ms。上游用 ONNX Runtime 会更快；这个耗时是托管内核的结果。

## 与上游的差别

- 不带 `magika-rules` / YARA。只做模型、阈值、overwrite map，以及空文件和短文本启发式。`OverwriteReason.RulesVeto` 保留给以后，当前不会出现。
- 内容类型表仍包含仅由规则产生的标签（例如 `empty`、`directory`），方便和上游的标签集对照。
- 目标框架是 `net11.0`，库标记了 `IsAotCompatible`。

## 开发

需要 .NET 11 SDK（`global.json` 固定到 11.0 预览通道，允许向前滚动）。

```bash
dotnet test Magika.slnx -c Release
```

测试对照一组小样本（PDF、PNG、ZIP、文本、JavaScript、JSON、HTML、Python、gzip、ELF）的标签，并把概率与 ONNX Runtime 的输出比较。标签在两个系统上都必须一致。Linux 上概率最大绝对误差不超过 1e-3；Windows 允许 0.08，用来盖住检出换行和 libm 造成的尖峰漂移。样本在 `.gitattributes` 里标成 binary，避免 Windows 把 LF 改成 CRLF。ONNX Runtime 只用于生成那份对照，不是运行时依赖。

GitHub Actions 在 `ubuntu-latest` 和 `windows-latest` 上跑同一条 `dotnet test`。推送 `v*` tag（当前版本是 `v0.1.1`）或手动跑 Release workflow 时，会用 Native AOT 打出 `linux-x64`、`osx-arm64` 和 `win-x64` 的 zip，并附上 `SHA256SUMS.txt`。`osx-arm64` 在 `macos-latest`（Apple Silicon）上构建并链接。每个 zip 里有 `RUN.txt`。

## English

Magika.NET identifies file content types with Google's Magika `standard_v3_3` model on .NET 11. `MagikaSession.Identify` accepts a path, a stream, or bytes and returns the label, MIME type, group, score, and overwrite reason used by upstream Magika (per-label thresholds, overwrite map, beginning/end tokens).

ONNX runs in-process. The interpreter is original C# in the style of SimdPaddleOCR: handwritten AVX-512 and AVX2/FMA GEMM, no ONNX Runtime and no native inference library. Other CPUs, including Apple Silicon, use `System.Numerics.Vector<float>`. SimdPaddleOCR itself is not referenced; its OCR operator set does not cover this graph. See `THIRD-PARTY-NOTICES.md` for the Google model license and that design credit.

The CLI project is `src/Magika.Cli`. Its assembly name is `Magika.Cli` and the tool command name is `magika`. This build does not ship Magika's YARA rules. A `v*` tag such as `v0.1.1` publishes Native AOT zips for `linux-x64`, `osx-arm64`, and `win-x64`. The `osx-arm64` binary is built on `macos-latest`.
