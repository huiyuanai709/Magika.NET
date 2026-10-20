**English** | [中文](README.zh-CN.md)

# Magika.NET

AI content-type detection for .NET 11. Magika.NET uses Google's bundled Magika `standard_v3_3` model to identify a file (PDF, PNG, ZIP, ELF, text, and hundreds more). Inference is pure managed ONNX, without Microsoft.ML.OnnxRuntime or a native inference library.

## Library

Reference `src/Magika/Magika.csproj`, or pack the library as `Magika.NET` (`net11.0`).

```csharp
using Magika;

MagikaSession session = MagikaSession.Create();
MagikaResult result = session.Identify(File.ReadAllBytes(path));

Console.WriteLine(result.ContentType.Label);     // pdf
Console.WriteLine(result.ContentType.MimeType);  // application/pdf
Console.WriteLine(result.ContentType.Group);     // document
Console.WriteLine(result.Score);
```

`Identify` accepts bytes. `Identify(Stream)` restores the stream position and, when the stream can seek, reads only a block from each end. `IdentifyPath` reports a directory as `directory`. Symlinks are followed unless `FollowSymlinks` is false, which reports `symlink`.

`IdentifyAsync` and `IdentifyPathAsync` take a `CancellationToken` and read those same prefix and suffix windows. Paths use `RandomAccess.ReadAsync`. The library and CLI compile async methods with runtime async (`<Features>$(Features);runtime-async=on</Features>`). Cancellation stops the read; the ONNX run itself is synchronous.

`IdentifyPaths` and `IdentifyPathsAsync` keep input order. They read paths in windows of 64 and score a window while the next one is read. Model rows are scored in batches (`MagikaOptions.BatchSize`, default 8) on a pool of managed sessions (`InferenceWorkers`). `IdentifyFeatures` and `IdentifyFeaturesBatch` score vectors you already extracted (2048 integers per row for this model). One session is safe to share across threads: single-file calls take one lane, and the multi-file APIs use the pool. The pool is created on the first multi-file call. Lanes share the model weights and keep separate activation scratch. A lane runs one forward pass at a time. With `InferenceWorkers` left at 0, the specialized model uses one lane and lets the convolution occupy every processor. One lane per core is slower once that convolution is itself parallel, because the lanes then compete for the same cores and for cache. An interpreter graph still uses one lane per processor. An explicit lane count is used as given and is not capped; extra lanes split the convolution's threads so the product stays near the processor count.

```csharp
MagikaSession session = MagikaSession.Create(new MagikaOptions
{
    PredictionMode = PredictionMode.MediumConfidence,
    FollowSymlinks = false,
});
```

| Mode | Behavior |
| --- | --- |
| `HighConfidence` (default) | Keep the label only when its own high-confidence threshold is met |
| `MediumConfidence` | Keep the label when the score meets the model medium threshold (`0.5` for `standard_v3_3`) |
| `BestGuess` | Always keep the model's best label (the overwrite map still applies) |

Below that threshold, a text-like label becomes `txt` and any other label becomes `unknown` (`OverwriteReason.LowConfidence`). The overwrite map turns `randombytes` into `unknown` and `randomtxt` into `txt`. An empty file is `empty`. A short input that passes a UTF-8 check is `txt`; otherwise it is `unknown`, and the model does not run.

## Command line

The CLI project is `src/Magika.Cli`. Its assembly name is `Magika.Cli` and the command name is `magika`.

```bash
dotnet run --project src/Magika.Cli -c Release -- sample.pdf
dotnet run --project src/Magika.Cli -c Release -- -r --prediction-mode medium --no-dereference ./files
dotnet run --project src/Magika.Cli -c Release -- --jsonl -s ./files
dotnet run --project src/Magika.Cli -c Release -- --batch-size 4 --workers 2 ./files/*
```

The flags match the upstream Magika CLI. `-r` / `--recursive` identifies files inside directories. Without it, a directory is reported as `directory`. `-` reads standard input once. `--no-dereference` reports a symlink as `symlink`. `--json` prints a pretty JSON array and `--jsonl` prints one compact JSON object per line, both in the upstream `path` / `result.status` / `result.value` shape (`dl`, `output`, and `score`). `-s` adds a truncated percent score, `-i` prints the MIME type, and `-l` prints the label. `--format` accepts `%p %l %d %g %m %e %s %S %b %%`. Those modifiers cannot be combined with `--json`, `--jsonl`, or `--format`, and `-i` cannot be combined with `-l`. `--colors` and `--no-colors` are mutually exclusive. Colors are used when stdout is a terminal, unless `NO_COLOR` is set.

The default line is `path: description (group)`, the same shape as upstream. The previous Magika.NET line was `path: label score mime group`. `--format "%p: %l %s %m %g"` is the close equivalent (`%s` prints two digits). JSON scores are truncated to three decimal places. Per-file errors stay in the output (and in JSON as `file_does_not_exist`, `permission_error`, `directory_cycle`, or `unknown`) and the process exits `1` after the remaining paths. Exit code `2` means the arguments were invalid.

If an argument contains `*` or `?` and that exact path does not exist, the CLI expands it as a file-name pattern. An existing path is kept as written. One content path uses `IdentifyPathAsync`. Several paths are read concurrently and scored with `IdentifyPathsAsync`. `--batch-size` (default 8, range 1..64) is how many feature rows share a forward pass. `--workers` (default 0) is the inference pool. Automatic mode uses one lane for this specialized model so the convolution can use every processor, and one lane per processor for an interpreter graph. An explicit count is used as given, with no cap. Each lane is a managed session over the same weights. Extra lanes split the convolution's thread budget. This stays on the managed interpreter.

`-V` / `--version` prints the library version and the model name. A binary published with Satori GC adds a second line, `gc: satori`.

A release binary is `./magika` on Linux and macOS, and `magika.exe` on Windows. See `RUN.txt` inside each zip.

## Model

The embedded model is Google's [`standard_v3_3`](https://github.com/google/magika/tree/main/assets/models/standard_v3_3) (214 labels). `model.onnx` and `config.min.json` are resources in the `Magika` assembly. Source and copyright notes are in `assets/models/standard_v3_3/ORIGIN.txt`.

Input tokens follow upstream Magika: up to 4096 bytes from the start and from the end, with whitespace at those edges removed, formed into 1024 + 1024 integers and padded with `256`. This model has no middle window. YARA rules are not included.

## Inference

The graph runs in-process (`src/Magika/Onnx`). x64 uses handwritten AVX-512 and AVX2/FMA kernels. Apple Silicon uses AdvSimd (`Vector128`) on the same dense, softmax, layer-norm, and GELU helpers, and `System.Numerics.Vector<float>` for the portable convolution. Those managed kernels are inspired by [SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR). Framework-dependent runs pick the kernel at runtime. Native AOT freezes that choice, so the release zips below are separate AVX2 and AVX-512 binaries. The AVX-512 zip also sets `IlcMaxVectorTBitWidth` to 512 so `System.Numerics.Vector<T>` is 512 bits wide. The AVX2 zip leaves that width at the x64 default. The macOS zip is one binary built with `IlcInstructionSet=apple-m1`, which keeps AdvSimd.

`OnnxSession.Run` evaluates a real batch (the leading dimension is the row count). A session keeps the immutable initializers in one dictionary and reuses activation buffers for the same shapes, so a warm run does not allocate those tensors again. The returned probabilities are a copy, so the next run can reuse the output buffer. There is still no Microsoft.ML.OnnxRuntime dependency.

## Releases

Project version `0.1.3`. Pushing a `v*` tag such as `v0.1.3`, or running the Release workflow with that version, publishes Native AOT zips and `SHA256SUMS.txt`:

| Asset | Built on | CPU |
| --- | --- | --- |
| `magika-<tag>-linux-x64-avx2.zip` | `ubuntu-latest` | x64 with AVX2 |
| `magika-<tag>-linux-x64-avx512.zip` | `ubuntu-latest` | x64 with AVX-512 |
| `magika-<tag>-win-x64-avx2.zip` | `windows-latest` | x64 with AVX2 |
| `magika-<tag>-win-x64-avx512.zip` | `windows-latest` | x64 with AVX-512 |
| `magika-<tag>-osx-arm64.zip` | `macos-latest` (Apple Silicon) | arm64, apple-m1 / AdvSimd |

Each zip contains the binary, `RUN.txt`, `LICENSE`, `THIRD-PARTY-NOTICES.md`, and `VERSION.txt`.

Those Native AOT binaries use [Satori GC](https://www.nuget.org/packages/PublishWithSatoriGC/11.0.0-satori.38707303131.1) (MIT, .NET Foundation). `src/Magika.Cli` imports the `PublishWithSatoriGC` SDK at `11.0.0-satori.38707303131.1`. `UseSatoriGC` stays at the SDK default, `true`, so a self-contained or `PublishAot` publish of the CLI uses Satori. `dotnet pack` still produces a framework-dependent `PackAsTool` package; Satori applies to the self-contained Native AOT release zips, not to that tool pack.

## Development

The SDK is .NET 11. `global.json` selects the 11.0 preview channel and allows roll-forward.

```bash
dotnet test Magika.slnx -c Release
```

GitHub Actions runs that test on `ubuntu-latest` and `windows-latest`.

## License

Magika.NET is [Apache-2.0](LICENSE). The embedded `standard_v3_3` model and the content-type catalog derived from Google Magika stay under Google's copyright and Apache-2.0. Details are in `THIRD-PARTY-NOTICES.md`.
