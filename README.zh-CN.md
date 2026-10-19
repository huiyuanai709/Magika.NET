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

`IdentifyAsync` 和 `IdentifyPathAsync` 接受 `CancellationToken`，读取的仍是同一段前缀和后缀。路径使用 `RandomAccess.ReadAsync`。库和 CLI 用 `<Features>$(Features);runtime-async=on</Features>` 编译异步方法。取消只作用于读取；ONNX 推理本身是同步的。

`IdentifyPaths` 和 `IdentifyPathsAsync` 保持输入顺序。它们按 64 个路径为一窗读取，打分当前窗时继续读下一窗。模型行按批打分（`MagikaOptions.BatchSize`，默认 8），用的是一池托管会话（`InferenceWorkers`）。`IdentifyFeatures` 和 `IdentifyFeaturesBatch` 给已经抽好的向量打分（这个模型每行 2048 个整数）。一个会话可以跨线程共用：单文件调用占用一条推理通道，多文件 API 使用通道池。通道池在第一次多文件调用时创建。各通道共享模型权重，激活暂存各自独立。一条通道同一时间只跑一次前向。`InferenceWorkers` 为 0 时，专用模型只用一条通道，卷积占用全部处理器。专用卷积本身已经并行，再按核数开通道会抢核、抢缓存。仍走解释器的图则还是每个处理器一条通道。显式指定的通道数按原值使用，不设上限；多出来的通道会把卷积线程拆开，使通道数乘每通道线程数接近处理器数量。

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
dotnet run --project src/Magika.Cli -c Release -- --batch-size 4 --workers 2 ./files/*
```

文本输出是 `路径: 标签 分数 MIME 分组`，顺序与参数一致。单个路径走 `IdentifyPathAsync`。多个路径并发读取，再用 `IdentifyPathsAsync` 打分。`--batch-size`（默认 8，范围 1..64）是一次前向里的特征行数。`--workers`（默认 0）是推理池。自动模式对这个专用模型只用一条通道，卷积可以使用全部处理器；解释器图则每个处理器一条通道。显式给出的数量会按原值使用，不设上限。每条通道是同一份权重上的托管会话。增加通道会把卷积的线程预算拆开。推理仍然是托管解释器。

`--version` 打印库版本和模型名。用 Satori GC 发布的二进制会多打一行 `gc: satori`。退出码 `1` 表示文件读不到。退出码 `2` 表示参数无效。

发布包里的程序，Linux 和 macOS 是 `./magika`，Windows 是 `magika.exe`。用法写在每个 zip 的 `RUN.txt` 里。

## 模型

内置模型是 Google 的 [`standard_v3_3`](https://github.com/google/magika/tree/main/assets/models/standard_v3_3)（214 类）。`model.onnx` 和 `config.min.json` 嵌在 `Magika` 程序集里。来源和版权说明在 `assets/models/standard_v3_3/ORIGIN.txt`。

输入 token 与上游 Magika 一致：从文件开头和结尾各取最多 4096 字节，去掉这两端的空白，组成 1024 + 1024 个整数，不足处用 `256` 填充。这个模型没有中间窗口。不包含 YARA 规则。

## 推理

图在进程内执行（`src/Magika/Onnx`）。x64 使用手写的 AVX-512 和 AVX2/FMA 内核。其它 CPU（包括 Apple Silicon）使用 `System.Numerics.Vector<float>`。这些托管内核的写法受到 [SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR) 的启发。框架依赖的运行会在运行时选择内核。Native AOT 会把这个选择固定下来，因此下面的发布 zip 把 AVX2 和 AVX-512 分成两份二进制。AVX-512 包还会把 `IlcMaxVectorTBitWidth` 设为 512，让 `System.Numerics.Vector<T>` 使用 512 位。AVX2 包保持 x64 上的默认宽度。

`OnnxSession.Run` 按真实 batch 计算（第一维是行数）。会话把不可变的 initializer 放在同一个字典里，并按形状复用激活缓冲，因此热运行不会再分配这些张量。返回的概率是一份拷贝，下一次运行可以复用输出缓冲。仍然不依赖 Microsoft.ML.OnnxRuntime。

## 发布

项目版本 `0.1.2`。推送 `v0.1.2` 这样的 `v*` tag，或用该版本手动跑 Release workflow，会发布 Native AOT zip 和 `SHA256SUMS.txt`：

| 文件 | 构建环境 | CPU |
| --- | --- | --- |
| `magika-<tag>-linux-x64-avx2.zip` | `ubuntu-latest` | 带 AVX2 的 x64 |
| `magika-<tag>-linux-x64-avx512.zip` | `ubuntu-latest` | 带 AVX-512 的 x64 |
| `magika-<tag>-win-x64-avx2.zip` | `windows-latest` | 带 AVX2 的 x64 |
| `magika-<tag>-win-x64-avx512.zip` | `windows-latest` | 带 AVX-512 的 x64 |
| `magika-<tag>-osx-arm64.zip` | `macos-latest`（Apple Silicon） | arm64 |

每个 zip 里有二进制、`RUN.txt`、`LICENSE`、`THIRD-PARTY-NOTICES.md` 和 `VERSION.txt`。

这些 Native AOT 二进制使用 [Satori GC](https://www.nuget.org/packages/PublishWithSatoriGC/11.0.0-satori.38707303131.1)（MIT，.NET Foundation）。`src/Magika.Cli` 引用 `PublishWithSatoriGC` SDK，版本钉在 `11.0.0-satori.38707303131.1`。`UseSatoriGC` 保持 SDK 默认值 `true`，因此 CLI 的自包含或 `PublishAot` 发布会换上 Satori。`dotnet pack` 仍然打出框架依赖的 `PackAsTool` 包；Satori 只用于自包含的 Native AOT 发布 zip，不进入该工具包。

## 开发

需要 .NET 11 SDK。`global.json` 选择 11.0 预览通道，并允许向前滚动。

```bash
dotnet test Magika.slnx -c Release
```

GitHub Actions 在 `ubuntu-latest` 和 `windows-latest` 上跑同一条测试。

## 许可

Magika.NET 使用 [Apache-2.0](LICENSE)。内置的 `standard_v3_3` 模型，以及从 Google Magika 整理的内容类型表，仍属于 Google 的版权并使用 Apache-2.0。详见 `THIRD-PARTY-NOTICES.md`。
