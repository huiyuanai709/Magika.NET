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
dotnet run --project src/Magika.Cli -c Release -- --json --prediction-mode medium --no-dereference ./files
```

Text output is `path: label score mime group`. `--version` prints the library version and the model name. A binary published with Satori GC adds a second line, `gc: satori`. Exit code `1` means a file could not be read. Exit code `2` means the arguments were invalid.

A release binary is `./magika` on Linux and macOS, and `magika.exe` on Windows. See `RUN.txt` inside each zip.

## Model

The embedded model is Google's [`standard_v3_3`](https://github.com/google/magika/tree/main/assets/models/standard_v3_3) (214 labels). `model.onnx` and `config.min.json` are resources in the `Magika` assembly. Source and copyright notes are in `assets/models/standard_v3_3/ORIGIN.txt`.

Input tokens follow upstream Magika: up to 4096 bytes from the start and from the end, with whitespace at those edges removed, formed into 1024 + 1024 integers and padded with `256`. This model has no middle window. YARA rules are not included.

## Inference

The graph runs in-process (`src/Magika/Onnx`). x64 uses handwritten AVX-512 and AVX2/FMA kernels. Other CPUs, including Apple Silicon, use `System.Numerics.Vector<float>`. Those managed kernels are inspired by [SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR).

## Releases

Project version `0.1.1`. Pushing a `v*` tag such as `v0.1.1`, or running the Release workflow with that version, publishes Native AOT zips and `SHA256SUMS.txt`:

| Asset | Built on |
| --- | --- |
| `magika-<tag>-linux-x64.zip` | `ubuntu-latest` |
| `magika-<tag>-osx-arm64.zip` | `macos-latest` (Apple Silicon) |
| `magika-<tag>-win-x64.zip` | `windows-latest` |

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
