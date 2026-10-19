[English](README.md) | **中文**

# Magika.NET

面向 .NET 11 的 AI 内容类型识别。Magika.NET 使用内置的 Google Magika `standard_v3_3` 模型判断文件类型（PDF、PNG、ZIP、ELF、文本，以及更多类型）。推理是纯托管 ONNX，不使用 Microsoft.ML.OnnxRuntime，也没有本机推理库。

## 库

引用 `src/Magika/Magika.csproj`，或把库打包为 `Magika.NET`（`net11.0`）。

```csharp
using Magika;

MagikaSession session = MagikaSession.Create();
MagikaResult result = session.Identify(File.ReadAllBytes(path));

Console.WriteLine(result.ContentType.Label);     // pdf
Console.WriteLine(result.ContentType.MimeType);  // application/pdf
Console.WriteLine(result.ContentType.Group);     // document
Console.WriteLine(result.Score);
```

`Identify` 接受字节。`Identify(Stream)` 会恢复流的位置；流可查找时，只从开头和结尾各读一块。`IdentifyPath` 把目录标成 `directory`。符号链接默认跟随目标；`FollowSymlinks` 为 false 时标成 `symlink`。

```csharp
MagikaSession session = MagikaSession.Create(new MagikaOptions
{
    PredictionMode = PredictionMode.MediumConfidence,
    FollowSymlinks = false,
});
```

| 模式 | 行为 |
| --- | --- |
| `HighConfidence`（默认） | 分数达到该标签自己的高置信阈值才保留 |
| `MediumConfidence` | 分数达到模型的中等阈值才保留（`standard_v3_3` 为 `0.5`） |
| `BestGuess` | 始终保留模型的最佳标签（仍会走 overwrite map） |

低于该阈值时，文本类标签变成 `txt`，其它标签变成 `unknown`（`OverwriteReason.LowConfidence`）。overwrite map 把 `randombytes` 改成 `unknown`，把 `randomtxt` 改成 `txt`。空文件是 `empty`。能通过 UTF-8 检查的短内容是 `txt`，否则是 `unknown`，并且不跑模型。

## 命令行

命令行项目是 `src/Magika.Cli`。程序集名称是 `Magika.Cli`，命令名是 `magika`。

```bash
dotnet run --project src/Magika.Cli -c Release -- sample.pdf
dotnet run --project src/Magika.Cli -c Release -- --json --prediction-mode medium --no-dereference ./files
```

文本输出是 `路径: 标签 分数 MIME 分组`。`--version` 打印库版本和模型名。退出码 `1` 表示文件读不到。退出码 `2` 表示参数无效。

发布包里的程序，Linux 和 macOS 是 `./magika`，Windows 是 `magika.exe`。用法写在每个 zip 的 `RUN.txt` 里。

## 模型

内置模型是 Google 的 [`standard_v3_3`](https://github.com/google/magika/tree/main/assets/models/standard_v3_3)（214 类）。`model.onnx` 和 `config.min.json` 嵌在 `Magika` 程序集里。来源和版权说明在 `assets/models/standard_v3_3/ORIGIN.txt`。

输入 token 与上游 Magika 一致：从文件开头和结尾各取最多 4096 字节，去掉这两端的空白，组成 1024 + 1024 个整数，不足处用 `256` 填充。这个模型没有中间窗口。不包含 YARA 规则。

## 推理

图在进程内执行（`src/Magika/Onnx`）。x64 使用手写的 AVX-512 和 AVX2/FMA 内核。其它 CPU（包括 Apple Silicon）使用 `System.Numerics.Vector<float>`。这些托管内核的写法受到 [SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR) 的启发。

## 发布

项目版本 `0.1.1`。推送 `v0.1.1` 这样的 `v*` tag，或用该版本手动跑 Release workflow，会发布 Native AOT zip 和 `SHA256SUMS.txt`：

| 文件 | 构建环境 |
| --- | --- |
| `magika-<tag>-linux-x64.zip` | `ubuntu-latest` |
| `magika-<tag>-osx-arm64.zip` | `macos-latest`（Apple Silicon） |
| `magika-<tag>-win-x64.zip` | `windows-latest` |

每个 zip 里有二进制、`RUN.txt`、`LICENSE`、`THIRD-PARTY-NOTICES.md` 和 `VERSION.txt`。

## 开发

需要 .NET 11 SDK。`global.json` 选择 11.0 预览通道，并允许向前滚动。

```bash
dotnet test Magika.slnx -c Release
```

GitHub Actions 在 `ubuntu-latest` 和 `windows-latest` 上跑同一条测试。

## 许可

Magika.NET 使用 [Apache-2.0](LICENSE)。内置的 `standard_v3_3` 模型，以及从 Google Magika 整理的内容类型表，仍属于 Google 的版权并使用 Apache-2.0。详见 `THIRD-PARTY-NOTICES.md`。
