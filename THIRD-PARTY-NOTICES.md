# Third-party notices

Magika.NET is licensed under the Apache License, Version 2.0. The components below are included or referenced under their own terms.

## Google Magika

- Project: https://github.com/google/magika
- License: Apache License, Version 2.0
- Copyright: Copyright 2024 Google LLC

The embedded `standard_v3_3` files `model.onnx`, `config.min.json`, and `metadata.json` are Google's published model assets (see `assets/models/standard_v3_3/ORIGIN.txt`). The content-type catalog in `src/Magika/ContentTypes.g.cs` is derived from Magika's `rust/lib/src/content.rs`. Feature extraction, confidence thresholds, and the overwrite map follow Magika's Rust library. These materials remain under Google's copyright and the Apache License, Version 2.0.

Magika.NET does not include the upstream `magika-rules` / YARA rule engine.

## SimdPaddleOCR (design reference)

- Project: https://github.com/sdcb/SimdPaddleOCR
- Mirror: https://github.com/huiyuanai709/SimdPaddleOCR
- License: Apache License, Version 2.0

No SimdPaddleOCR source, package, or native binary is vendored in this repository. Magika.NET's ONNX interpreter and GEMM are original code. The design follows that project's approach: a pure C# ONNX interpreter with handwritten CPU kernels, and no dependency on ONNX Runtime or Paddle Inference. SimdPaddleOCR's interpreter is built for OCR graphs and does not implement operators this model needs (`Equal`, `Cast`, `Shape`, `Expand`, `Reciprocal`, `GlobalMaxPool`), so the stack was not reused as a package or a source fork.
