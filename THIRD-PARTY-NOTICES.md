# Third-party notices

Magika.NET is licensed under the Apache License, Version 2.0. The components below are included or referenced under their own terms.

## Google Magika

- Project: https://github.com/google/magika
- License: Apache License, Version 2.0
- Copyright: Copyright 2024 Google LLC

The embedded `standard_v3_3` files `model.onnx`, `config.min.json`, and `metadata.json` are Google's published model assets (see `assets/models/standard_v3_3/ORIGIN.txt`). The content-type catalog in `src/Magika/ContentTypes.g.cs` is derived from Magika's `rust/lib/src/content.rs`. Feature extraction, confidence thresholds, and the overwrite map follow Magika's Rust library. These materials remain under Google's copyright and the Apache License, Version 2.0.

Magika.NET does not include the upstream `magika-rules` / YARA rule engine.

## Satori GC

- Package: https://www.nuget.org/packages/PublishWithSatoriGC/11.0.0-satori.38707303131.1
- Project: https://github.com/VSadov/Satori
- License: MIT License
- Copyright: Copyright (c) .NET Foundation and Contributors

Native AOT release builds of `Magika.Cli` publish with Satori GC through the `PublishWithSatoriGC` SDK (`11.0.0-satori.38707303131.1`). The SDK and its runtime packs are not vendored in this repository. The framework-dependent `PackAsTool` package does not include them. The MIT license text is the license distributed with that NuGet package.

## SimdPaddleOCR (design reference)

- Project: https://github.com/sdcb/SimdPaddleOCR
- Mirror: https://github.com/huiyuanai709/SimdPaddleOCR
- License: Apache License, Version 2.0

No SimdPaddleOCR source, package, or native binary is vendored in this repository. Magika.NET's ONNX interpreter and GEMM are original code. The design follows that project's approach: a pure C# ONNX interpreter with handwritten CPU kernels, and no dependency on ONNX Runtime or Paddle Inference. SimdPaddleOCR's interpreter is built for OCR graphs and does not implement operators this model needs (`Equal`, `Cast`, `Shape`, `Expand`, `Reciprocal`, `GlobalMaxPool`), so the stack was not reused as a package or a source fork.
