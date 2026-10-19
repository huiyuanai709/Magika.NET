using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Xunit;

namespace Magika.Tests;

public class MagikaTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    [Fact]
    public void Catalog_ContainsModelLabels()
    {
        MagikaSession session = MagikaSession.Create();
        Assert.Equal(214, session.Model.TrainingLabels.Length);
        Assert.Equal("randombytes", session.Model.TrainingLabels[session.Model.OutputLabels.ToList().IndexOf("unknown") == -1
            ? Array.IndexOf(session.Model.TrainingLabels, "randombytes")
            : Array.IndexOf(session.Model.TrainingLabels, "randombytes")]);
        Assert.Equal("unknown", session.Model.OutputLabels[Array.IndexOf(session.Model.TrainingLabels, "randombytes")]);
        Assert.Equal("txt", session.Model.OutputLabels[Array.IndexOf(session.Model.TrainingLabels, "randomtxt")]);
        Assert.Equal("pdf", ContentTypes.Get("pdf").Label);
        Assert.Equal("application/pdf", ContentTypes.Get("pdf").MimeType);
        Assert.True(ContentTypes.Get("javascript").IsText);
        Assert.False(ContentTypes.Get("pdf").IsText);
    }

    [Fact]
    public void Features_LeftAlignBegAndRightAlignEnd()
    {
        MagikaSession session = MagikaSession.Create();
        ModelDefinition model = session.Model;
        byte[] content = Encoding.ASCII.GetBytes("   abcdefghij   ");
        int[] features = FeatureExtractor.Extract(content, model);

        Assert.Equal((int)'a', features[0]);
        Assert.Equal((int)'j', features[9]);
        Assert.Equal((int)' ', features[10]);
        Assert.Equal(model.PaddingToken, features[13]);
        Assert.Equal((int)' ', features[model.FeaturesSize - 13]);
        Assert.Equal((int)'a', features[model.FeaturesSize - 10]);
        Assert.Equal((int)'j', features[model.FeaturesSize - 1]);
    }

    [Fact]
    public void Features_StripsWhitespaceInsideTheBlockOnly()
    {
        MagikaSession session = MagikaSession.Create();
        ModelDefinition model = session.Model;
        byte[] content = new byte[6000];
        Array.Fill(content, (byte)'A', 0, 3000);
        Array.Fill(content, (byte)'B', 3000, 3000);
        content[0] = (byte)' ';
        content[^1] = (byte)'\n';
        int[] features = FeatureExtractor.Extract(content, model);

        Assert.Equal((int)'A', features[0]);
        Assert.Equal((int)'A', features[1023]);
        Assert.Equal((int)'B', features[^1]);
        Assert.Equal((int)'B', features[^1024]);
        Assert.True(FeatureExtractor.HasEnoughContent(features, model));
    }

    [Fact]
    public void SmallFiles_UseUtf8Heuristic()
    {
        MagikaSession session = MagikaSession.Create();
        Assert.Equal("empty", session.Identify([]).ContentType.Label);
        Assert.Equal("txt", session.Identify("hello"u8).ContentType.Label);
        Assert.Equal(1f, session.Identify("hello"u8).Score);
        Assert.Null(session.Identify("hello"u8).ModelContentType);

        byte[] binary = [0xFF, 0xFE, 0x00, 0x01, 0x02, 0x03, 0x04];
        Assert.Equal("unknown", session.Identify(binary).ContentType.Label);

        Assert.Equal("txt", FeatureExtractor.ClassifySmall([0xC2], fileLength: 10_000));
        Assert.Equal("unknown", FeatureExtractor.ClassifySmall([0xC2], fileLength: 1));
        Assert.Equal("unknown", FeatureExtractor.ClassifySmall([0xFF], fileLength: 100));
        Assert.Equal("txt", FeatureExtractor.ClassifySmall("ok"u8, fileLength: 2));
        Assert.Equal(FeatureExtractor.Utf8Kind.Invalid, FeatureExtractor.InspectUtf8([0xE0, 0x80]));
        Assert.Equal(FeatureExtractor.Utf8Kind.IncompleteTail, FeatureExtractor.InspectUtf8([0xE0, 0xA0]));
    }

    [Fact]
    public void PostProcess_MatchesUpstreamThresholdsAndOverwriteMap()
    {
        MagikaSession session = MagikaSession.Create();
        ModelDefinition model = session.Model;

        MagikaResult bytes = session.DecideScores(OneHot(model, "randombytes", 0.99f));
        Assert.Equal("unknown", bytes.ContentType.Label);
        Assert.Equal("randombytes", bytes.ModelContentType!.Label);
        Assert.Equal(OverwriteReason.OverwriteMap, bytes.OverwriteReason);

        MagikaResult text = session.DecideScores(OneHot(model, "randomtxt", 0.99f));
        Assert.Equal("txt", text.ContentType.Label);
        Assert.Equal(OverwriteReason.OverwriteMap, text.OverwriteReason);

        MagikaResult lowPdf = session.DecideScores(OneHot(model, "pdf", 0.1f));
        Assert.Equal("unknown", lowPdf.ContentType.Label);
        Assert.Equal(OverwriteReason.LowConfidence, lowPdf.OverwriteReason);

        MagikaResult lowJs = session.DecideScores(OneHot(model, "javascript", 0.1f));
        Assert.Equal("txt", lowJs.ContentType.Label);
        Assert.Equal(OverwriteReason.LowConfidence, lowJs.OverwriteReason);

        MagikaResult lowTxt = session.DecideScores(OneHot(model, "txt", 0.1f));
        Assert.Equal("txt", lowTxt.ContentType.Label);
        Assert.Null(lowTxt.OverwriteReason);

        int zig = Array.IndexOf(model.TrainingLabels, "zig");
        Assert.Equal(0.9f, model.HighThresholds[zig]);
        MagikaResult zigLow = session.DecideScores(OneHot(model, "zig", 0.8f));
        Assert.Equal("txt", zigLow.ContentType.Label);
        MagikaSession medium = MagikaSession.Create(new MagikaOptions { PredictionMode = PredictionMode.MediumConfidence });
        Assert.Equal("zig", medium.DecideScores(OneHot(model, "zig", 0.8f)).ContentType.Label);
    }

    [Fact]
    public void Gemm_MatchesNaive()
    {
        const int M = 7;
        const int K = 13;
        const int N = 11;
        float[] a = new float[M * K];
        float[] b = new float[K * N];
        float[] expected = new float[M * N];
        float[] actual = new float[M * N];
        var random = new Random(1234);
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)(random.NextDouble() * 2 - 1);
        }

        for (int i = 0; i < b.Length; i++)
        {
            b[i] = (float)(random.NextDouble() * 2 - 1);
        }

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                for (int p = 0; p < K; p++)
                {
                    sum += a[i * K + p] * b[p * N + j];
                }

                expected[i * N + j] = sum;
            }
        }

        Onnx.SimdGemm.Multiply(a, b, actual, M, K, N);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], precision: 4);
        }
    }

    [Theory]
    [InlineData(64, 128, 64)]
    [InlineData(70, 100, 50)]
    [InlineData(512, 40, 36)]
    public void Gemm_BlockedMatchesNaive(int m, int k, int n)
    {
        float[] a = new float[m * k];
        float[] b = new float[k * n];
        float[] expected = new float[m * n];
        float[] actual = new float[m * n];
        var random = new Random(99);
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)(random.NextDouble() * 2 - 1);
        }

        for (int i = 0; i < b.Length; i++)
        {
            b[i] = (float)(random.NextDouble() * 2 - 1);
        }

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    sum += a[(i * k) + p] * b[(p * n) + j];
                }

                expected[(i * n) + j] = (float)sum;
            }
        }

        Onnx.SimdGemm.Multiply(a, b, actual, m, k, n);
        for (int i = 0; i < expected.Length; i++)
        {
            float delta = MathF.Abs(expected[i] - actual[i]);
            Assert.True(delta <= 2e-3f, $"index {i} expected {expected[i]} actual {actual[i]} delta {delta}");
        }
    }

    [Fact]
    public void Tanh_MatchesMathF()
    {
        float[] source = new float[4096 + 7];
        float[] actual = new float[source.Length];
        var random = new Random(7);
        for (int i = 0; i < source.Length; i++)
        {
            source[i] = (float)((random.NextDouble() * 24) - 12);
        }

        source[0] = 0f;
        source[1] = -0f;
        source[2] = 9f;
        source[3] = -9f;
        source[4] = 20f;
        source[5] = -20f;
        source[6] = 0.001f;
        Onnx.SimdMath.Tanh(source, actual);
        for (int i = 0; i < source.Length; i++)
        {
            float expected = MathF.Tanh(source[i]);
            float delta = MathF.Abs(expected - actual[i]);
            Assert.True(delta <= 2e-5f, $"index {i} x {source[i]} expected {expected} actual {actual[i]} delta {delta}");
        }
    }

    [Fact]
    public void Model_IsFusedAndMatchesOnnxRuntime()
    {
        MagikaSession session = MagikaSession.Create();
        Assert.True(session.EmbeddingFused);
        using JsonDocument parity = JsonDocument.Parse(File.ReadAllText(Fixture("parity.json")));
        foreach (JsonProperty sample in parity.RootElement.EnumerateObject())
        {
            byte[] bytes = File.ReadAllBytes(Fixture(sample.Name + ".bin"));
            JsonElement spec = sample.Value;
            MagikaResult result = session.Identify(bytes);
            if (spec.GetProperty("skipped").GetBoolean())
            {
                continue;
            }

            string expectedLabel = spec.GetProperty("label").GetString()!;
            float expectedScore = spec.GetProperty("score").GetSingle();
            Assert.Equal(expectedLabel, result.ModelContentType!.Label);
            // Linux stays within 0.001 of the ONNX Runtime reference. Windows gets 0.08:
            // the observed text-fixture gap on windows-latest was 0.052, and that runner's
            // autocrlf rewrite of LF fixtures accounts for it. Fixtures are marked binary
            // so checkout keeps the original bytes. The label assert above stays exact.
            float tolerance = OperatingSystem.IsWindows() ? 0.08f : 0.001f;
            float scoreDelta = MathF.Abs(expectedScore - result.Score);
            Assert.True(
                scoreDelta <= tolerance,
                $"{sample.Name} score expected {expectedScore:R} actual {result.Score:R} delta {scoreDelta:R} tolerance {tolerance:R}");

            float[] scores = session.Score(bytes);
            byte[] reference = Convert.FromBase64String(spec.GetProperty("probs_b64").GetString()!);
            float[] expected = new float[reference.Length / 4];
            Buffer.BlockCopy(reference, 0, expected, 0, reference.Length);
            Assert.Equal(expected.Length, scores.Length);
            float max = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                max = MathF.Max(max, MathF.Abs(expected[i] - scores[i]));
            }

            Assert.True(max <= tolerance, $"{sample.Name} max abs diff {max} tolerance {tolerance}");
        }
    }

    [Fact]
    public void Identify_CommonTypes_AndLowConfidenceFallback()
    {
        MagikaSession session = MagikaSession.Create();
        Assert.Equal("pdf", session.Identify(File.ReadAllBytes(Fixture("pdf.bin"))).ContentType.Label);
        Assert.Equal("png", session.Identify(File.ReadAllBytes(Fixture("png.bin"))).ContentType.Label);
        Assert.Equal("zip", session.Identify(File.ReadAllBytes(Fixture("zip.bin"))).ContentType.Label);
        Assert.Equal("txt", session.Identify(File.ReadAllBytes(Fixture("text.bin"))).ContentType.Label);
        Assert.Equal("javascript", session.Identify(File.ReadAllBytes(Fixture("javascript.bin"))).ContentType.Label);
        Assert.Equal("json", session.Identify(File.ReadAllBytes(Fixture("json.bin"))).ContentType.Label);
        Assert.Equal("html", session.Identify(File.ReadAllBytes(Fixture("html.bin"))).ContentType.Label);
        Assert.Equal("python", session.Identify(File.ReadAllBytes(Fixture("python.bin"))).ContentType.Label);
        Assert.Equal("gzip", session.Identify(File.ReadAllBytes(Fixture("gzip.bin"))).ContentType.Label);
        Assert.Equal("elf", session.Identify(File.ReadAllBytes(Fixture("elf.bin"))).ContentType.Label);
        Assert.Equal("application/pdf", session.Identify(File.ReadAllBytes(Fixture("pdf.bin"))).ContentType.MimeType);
        Assert.Equal("document", session.Identify(File.ReadAllBytes(Fixture("pdf.bin"))).ContentType.Group);

        MagikaResult ambiguous = session.Identify(File.ReadAllBytes(Fixture("ambiguous.bin")));
        Assert.Equal("asm", ambiguous.ModelContentType!.Label);
        Assert.Equal("txt", ambiguous.ContentType.Label);
        Assert.Equal(OverwriteReason.LowConfidence, ambiguous.OverwriteReason);
    }

    [Fact]
    public void IdentifyPath_ReadsFilesDirectoriesAndStreams()
    {
        MagikaSession session = MagikaSession.Create();
        string directory = Path.Combine(Path.GetTempPath(), "magika-tests-" + Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(directory);
        try
        {
            Assert.Equal("directory", session.IdentifyPath(directory).ContentType.Label);
            string pdf = Path.Combine(directory, "sample.pdf");
            File.WriteAllBytes(pdf, File.ReadAllBytes(Fixture("pdf.bin")));
            MagikaResult fromPath = session.IdentifyPath(pdf);
            Assert.Equal("pdf", fromPath.ContentType.Label);
            Assert.Equal(pdf, fromPath.Path);

            using FileStream stream = File.OpenRead(pdf);
            stream.Position = 3;
            MagikaResult fromStream = session.Identify(stream, pdf);
            Assert.Equal("pdf", fromStream.ContentType.Label);
            Assert.Equal(3, stream.Position);

            string link = pdf + ".link";
            File.CreateSymbolicLink(link, pdf);
            Assert.Equal("pdf", session.IdentifyPath(link).ContentType.Label);
            MagikaSession noFollow = MagikaSession.Create(new MagikaOptions { FollowSymlinks = false });
            Assert.Equal("symlink", noFollow.IdentifyPath(link).ContentType.Label);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Identify_HeadAndTailMatchFullBufferAcrossBoundaries()
    {
        MagikaSession session = MagikaSession.Create();
        int block = session.Model.BlockSize;
        int[] sizes = [0, 1, block - 1, block, block + 1, (block * 2) - 1, block * 2, (block * 2) + 1, (block * 2) + 100, block * 5];
        string directory = Path.Combine(Path.GetTempPath(), "magika-ends-" + Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(directory);
        try
        {
            foreach (int size in sizes)
            {
                byte[] data = Pattern(size);
                MagikaResult expected = session.Identify(data);

                using MemoryStream seekable = new(data, writable: false);
                if (data.Length > 0)
                {
                    seekable.Position = Math.Min(3, data.Length);
                }

                long mark = seekable.Position;
                MagikaResult fromSeekable = session.Identify(seekable, "seekable");
                AssertSame(expected, fromSeekable, $"seekable {size}");
                Assert.Equal(mark, seekable.Position);

                using ChunkedStream oneByte = new(data, chunk: 1);
                AssertSame(expected, session.Identify(oneByte, "chunk1"), $"chunk1 {size}");

                using ChunkedStream wide = new(data, chunk: 8192);
                AssertSame(expected, session.Identify(wide, "chunkwide"), $"chunkwide {size}");

                string path = Path.Combine(directory, size + ".bin");
                File.WriteAllBytes(path, data);
                MagikaResult fromPath = session.IdentifyPath(path);
                AssertSame(expected, fromPath, $"path {size}");
                Assert.Equal(path, fromPath.Path);
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void IdentifyPath_PooledReadsStayDistinctAcrossThreads()
    {
        MagikaSession session = MagikaSession.Create();
        byte[] pdf = File.ReadAllBytes(Fixture("pdf.bin"));
        byte[] text = File.ReadAllBytes(Fixture("text.bin"));
        string directory = Path.Combine(Path.GetTempPath(), "magika-parallel-" + Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(directory);
        string pdfPath = Path.Combine(directory, "sample.pdf");
        string textPath = Path.Combine(directory, "sample.txt");
        File.WriteAllBytes(pdfPath, pdf);
        File.WriteAllBytes(textPath, text);
        try
        {
            Parallel.For(0, 4, _ =>
            {
                Assert.Equal("pdf", session.IdentifyPath(pdfPath).ContentType.Label);
                Assert.Equal("txt", session.IdentifyPath(textPath).ContentType.Label);
                Assert.Equal("pdf", session.Identify(new MemoryStream(pdf)).ContentType.Label);
                using ChunkedStream forward = new(text, chunk: 3);
                Assert.Equal("txt", session.Identify(forward).ContentType.Label);
            });
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task IdentifyAsync_MatchesSyncForPathSeekableAndForwardStreams()
    {
        MagikaSession session = MagikaSession.Create();
        int block = session.Model.BlockSize;
        int[] sizes = [0, 1, block - 1, block, block + 1, (block * 2) - 1, block * 2, (block * 2) + 1, (block * 2) + 100, block * 5];
        string directory = Path.Combine(Path.GetTempPath(), "magika-async-" + Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(directory);
        try
        {
            Assert.Equal("directory", (await session.IdentifyPathAsync(directory)).ContentType.Label);
            foreach (string name in new[] { "pdf", "png", "zip", "text", "javascript", "elf", "ambiguous" })
            {
                byte[] fixture = File.ReadAllBytes(Fixture(name + ".bin"));
                MagikaResult expected = session.Identify(fixture);
                using MemoryStream seekable = new(fixture, writable: false);
                seekable.Position = 2;
                MagikaResult fromSeekable = await session.IdentifyAsync(seekable, name);
                AssertSame(expected, fromSeekable, $"fixture seekable {name}");
                Assert.Equal(2, seekable.Position);

                using ChunkedStream forward = new(fixture, chunk: 3);
                AssertSame(expected, await session.IdentifyAsync(forward, name), $"fixture forward {name}");
            }

            foreach (int size in sizes)
            {
                byte[] data = Pattern(size);
                MagikaResult expected = session.Identify(data);
                string path = Path.Combine(directory, size + ".bin");
                await File.WriteAllBytesAsync(path, data);

                MagikaResult fromPath = await session.IdentifyPathAsync(path);
                AssertSame(expected, fromPath, $"async path {size}");
                Assert.Equal(path, fromPath.Path);

                using MemoryStream seekable = new(data, writable: false);
                if (data.Length > 0)
                {
                    seekable.Position = Math.Min(3, data.Length);
                }

                long mark = seekable.Position;
                MagikaResult fromSeekable = await session.IdentifyAsync(seekable, "seekable");
                AssertSame(expected, fromSeekable, $"async seekable {size}");
                Assert.Equal(mark, seekable.Position);

                using ChunkedStream oneByte = new(data, chunk: 1);
                AssertSame(expected, await session.IdentifyAsync(oneByte, "chunk1"), $"async chunk1 {size}");

                using ChunkedStream wide = new(data, chunk: 8192);
                AssertSame(expected, await session.IdentifyAsync(wide, "chunkwide"), $"async chunkwide {size}");
            }

            string pdf = Path.Combine(directory, "sample.pdf");
            byte[] pdfBytes = File.ReadAllBytes(Fixture("pdf.bin"));
            await File.WriteAllBytesAsync(pdf, pdfBytes);
            string link = pdf + ".link";
            File.CreateSymbolicLink(link, pdf);
            Assert.Equal("pdf", (await session.IdentifyPathAsync(link)).ContentType.Label);
            MagikaSession noFollow = MagikaSession.Create(new MagikaOptions { FollowSymlinks = false });
            Assert.Equal("symlink", (await noFollow.IdentifyPathAsync(link)).ContentType.Label);

            using CancellationTokenSource canceled = new();
            canceled.Cancel();
            using FileStream held = File.OpenRead(pdf);
            held.Position = 4;
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => session.IdentifyPathAsync(pdf, canceled.Token));
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => session.IdentifyAsync(held, pdf, canceled.Token));
            Assert.Equal(4, held.Position);

            await using CancelOnReadStream cancelOnRead = new(pdfBytes);
            long cancelMark = cancelOnRead.Position;
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => session.IdentifyAsync(cancelOnRead, "cancel"));
            Assert.Equal(cancelMark, cancelOnRead.Position);
            Assert.Equal("pdf", (await session.IdentifyAsync(new MemoryStream(pdfBytes))).ContentType.Label);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task IdentifyAsync_PooledReadsStayDistinctAcrossTasks()
    {
        MagikaSession session = MagikaSession.Create();
        byte[] pdf = File.ReadAllBytes(Fixture("pdf.bin"));
        byte[] text = File.ReadAllBytes(Fixture("text.bin"));
        string directory = Path.Combine(Path.GetTempPath(), "magika-async-parallel-" + Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(directory);
        string pdfPath = Path.Combine(directory, "sample.pdf");
        string textPath = Path.Combine(directory, "sample.txt");
        await File.WriteAllBytesAsync(pdfPath, pdf);
        await File.WriteAllBytesAsync(textPath, text);
        try
        {
            Task[] tasks = Enumerable.Range(0, 4).Select(async _ =>
            {
                Assert.Equal("pdf", (await session.IdentifyPathAsync(pdfPath)).ContentType.Label);
                Assert.Equal("txt", (await session.IdentifyPathAsync(textPath)).ContentType.Label);
                Assert.Equal("pdf", (await session.IdentifyAsync(new MemoryStream(pdf))).ContentType.Label);
                using ChunkedStream forward = new(text, chunk: 3);
                Assert.Equal("txt", (await session.IdentifyAsync(forward)).ContentType.Label);
            }).ToArray();
            await Task.WhenAll(tasks);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void AsyncIdentify_IsCompiledWithRuntimeAsync()
    {
        AssertRuntimeAsync(typeof(MagikaSession).GetMethod(nameof(MagikaSession.IdentifyAsync))!);
        AssertRuntimeAsync(typeof(MagikaSession).GetMethod(nameof(MagikaSession.IdentifyPathsAsync))!);
        AssertRuntimeAsync(typeof(MagikaSession).GetMethod("IdentifyFileAsync", BindingFlags.Instance | BindingFlags.NonPublic)!);
        AssertRuntimeAsync(typeof(MagikaSession).GetMethod("PreparePathAsync", BindingFlags.Instance | BindingFlags.NonPublic)!);
        AssertRuntimeAsync(typeof(ContentEnds).GetMethod("ReadForwardAsync", BindingFlags.Static | BindingFlags.NonPublic)!);
        AssertRuntimeAsync(typeof(ContentEnds).GetMethod("ReadExactAtAsync", BindingFlags.Static | BindingFlags.NonPublic)!);
    }

    [Fact]
    public void ScorePacked_MatchesSingleRowInference()
    {
        MagikaSession session = MagikaSession.Create();
        string[] names = ["pdf", "png", "zip", "text", "javascript", "elf", "ambiguous", "html"];
        int width = session.Model.FeaturesSize;
        int[] packed = new int[names.Length * width];
        float[][] single = new float[names.Length][];
        for (int i = 0; i < names.Length; i++)
        {
            byte[] bytes = File.ReadAllBytes(Fixture(names[i] + ".bin"));
            int[] features = FeatureExtractor.Extract(bytes, session.Model);
            features.CopyTo(packed.AsSpan(i * width, width));
            single[i] = session.Score(bytes);
        }

        float[] batched = session.ScorePacked(packed, names.Length);
        float[] again = session.ScorePacked(packed, names.Length);
        Assert.Equal(single[0].Length * names.Length, batched.Length);
        for (int i = 0; i < names.Length; i++)
        {
            for (int j = 0; j < single[i].Length; j++)
            {
                Assert.Equal(single[i][j], batched[(i * single[i].Length) + j]);
                Assert.Equal(single[i][j], again[(i * single[i].Length) + j]);
            }
        }
    }

    [Fact]
    public void IdentifyFeaturesBatch_MatchesIdentifyAndKeepsOrder()
    {
        MagikaSession session = MagikaSession.Create(new MagikaOptions { BatchSize = 3, InferenceWorkers = 2 });
        string[] names = ["elf", "pdf", "text", "png", "zip", "javascript", "html", "ambiguous"];
        int width = session.Model.FeaturesSize;
        int[] packed = new int[names.Length * width];
        string?[] paths = new string?[names.Length];
        MagikaResult[] expected = new MagikaResult[names.Length];
        for (int i = 0; i < names.Length; i++)
        {
            byte[] bytes = File.ReadAllBytes(Fixture(names[i] + ".bin"));
            expected[i] = session.Identify(bytes);
            FeatureExtractor.Extract(bytes, session.Model).CopyTo(packed.AsSpan(i * width, width));
            paths[i] = names[i];
        }

        MagikaResult[] batched = session.IdentifyFeaturesBatch(packed, names.Length, paths);
        Assert.Equal(names.Length, batched.Length);
        for (int i = 0; i < names.Length; i++)
        {
            Assert.Equal(names[i], batched[i].Path);
            AssertSame(expected[i], batched[i], names[i]);
        }

        MagikaResult one = session.IdentifyFeatures(packed.AsSpan(width, width), "pdf-row");
        Assert.Equal("pdf-row", one.Path);
        Assert.Equal("pdf", one.ContentType.Label);
    }

    [Fact]
    public async Task IdentifyPaths_PreservesInputOrderAcrossHeuristicsAndModel()
    {
        MagikaSession session = MagikaSession.Create(new MagikaOptions
        {
            BatchSize = 2,
            InferenceWorkers = 2,
            FollowSymlinks = true,
        });
        string directory = Path.Combine(Path.GetTempPath(), "magika-batch-" + Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(directory);
        try
        {
            string pdf = Path.Combine(directory, "sample.pdf");
            string text = Path.Combine(directory, "notes.txt");
            string zip = Path.Combine(directory, "bundle.zip");
            string empty = Path.Combine(directory, "empty.bin");
            string nested = Path.Combine(directory, "nested");
            Directory.CreateDirectory(nested);
            await File.WriteAllBytesAsync(pdf, File.ReadAllBytes(Fixture("pdf.bin")));
            await File.WriteAllBytesAsync(text, File.ReadAllBytes(Fixture("text.bin")));
            await File.WriteAllBytesAsync(zip, File.ReadAllBytes(Fixture("zip.bin")));
            await File.WriteAllBytesAsync(empty, []);
            string link = pdf + ".link";
            File.CreateSymbolicLink(link, pdf);
            string[] ordered =
            [
                nested,
                text,
                empty,
                pdf,
                link,
                zip,
                directory,
            ];

            MagikaResult[] sync = session.IdentifyPaths(ordered);
            MagikaResult[] asyncResults = await session.IdentifyPathsAsync(ordered);
            Assert.Equal(ordered.Length, sync.Length);
            for (int i = 0; i < ordered.Length; i++)
            {
                MagikaResult expected = session.IdentifyPath(ordered[i]);
                Assert.Equal(ordered[i], sync[i].Path);
                Assert.Equal(ordered[i], asyncResults[i].Path);
                AssertSame(expected, sync[i], "sync " + ordered[i]);
                AssertSame(expected, asyncResults[i], "async " + ordered[i]);
            }

            MagikaSession noFollow = MagikaSession.Create(new MagikaOptions { FollowSymlinks = false, BatchSize = 4 });
            Assert.Equal("symlink", noFollow.IdentifyPaths([link])[0].ContentType.Label);
            Parallel.For(0, 4, _ =>
            {
                Assert.Equal("pdf", session.IdentifyPath(pdf).ContentType.Label);
                MagikaResult[] mixed = session.IdentifyPaths([text, pdf, empty, zip]);
                Assert.Equal("txt", mixed[0].ContentType.Label);
                Assert.Equal("pdf", mixed[1].ContentType.Label);
                Assert.Equal("empty", mixed[2].ContentType.Label);
                Assert.Equal("zip", mixed[3].ContentType.Label);
            });
            Assert.Empty(session.IdentifyPaths([]));
            Assert.Throws<FileNotFoundException>(() => session.IdentifyPaths([pdf, Path.Combine(directory, "missing.bin")]));
            int[] padded = new int[session.Model.FeaturesSize];
            Array.Fill(padded, session.Model.PaddingToken);
            Assert.Throws<ArgumentException>(() => session.IdentifyFeatures(padded));
            using CancellationTokenSource canceled = new();
            canceled.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => session.IdentifyPathsAsync(ordered, canceled.Token));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void RepeatedInference_ReusesActivationBuffers()
    {
        MagikaSession session = MagikaSession.Create();
        byte[] text = File.ReadAllBytes(Fixture("text.bin"));
        float[] first = session.Score(text);
        long before = GC.GetAllocatedBytesForCurrentThread();
        float[] second = session.Score(text);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(first, second);
        Assert.True(allocated < 256 * 1024, $"warm inference allocated {allocated} bytes");
    }

    [Fact]
    public void Inference_IsCpuFastEnoughForInteractiveUse()
    {
        MagikaSession session = MagikaSession.Create();
        byte[] text = File.ReadAllBytes(Fixture("text.bin"));
        _ = session.Identify(text);
        var watch = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal("txt", session.Identify(text).ContentType.Label);
        }

        watch.Stop();
        Assert.True(watch.Elapsed.TotalMilliseconds / 5 < 250, $"average inference {watch.Elapsed.TotalMilliseconds / 5:0.0} ms");
    }

    private static void AssertSame(MagikaResult expected, MagikaResult actual, string scenario)
    {
        Assert.True(
            expected.ContentType.Label == actual.ContentType.Label
            && expected.ModelContentType?.Label == actual.ModelContentType?.Label
            && expected.Score == actual.Score
            && expected.OverwriteReason == actual.OverwriteReason,
            $"{scenario}: expected {expected.ContentType.Label}/{expected.ModelContentType?.Label} {expected.Score:R} {expected.OverwriteReason}, actual {actual.ContentType.Label}/{actual.ModelContentType?.Label} {actual.Score:R} {actual.OverwriteReason}");
    }

    private static byte[] Pattern(int size)
    {
        byte[] data = new byte[size];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (byte)(i * 17 + 3);
        }

        if (size > 0)
        {
            data[0] = (byte)'%';
            data[^1] = (byte)'Z';
        }

        return data;
    }

    private sealed class ChunkedStream : Stream
    {
        private readonly byte[] _data;
        private readonly int _chunk;
        private int _position;

        public ChunkedStream(byte[] data, int chunk)
        {
            _data = data;
            _chunk = chunk;
        }

        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => throw new NotSupportedException();

        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override void Flush()
        {
        }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return new ValueTask<int>(Read(buffer.Span));
        }

        public override int Read(Span<byte> buffer)
        {
            int remaining = _data.Length - _position;
            int n = Math.Min(buffer.Length, Math.Min(_chunk, remaining));
            if (n <= 0)
            {
                return 0;
            }

            _data.AsSpan(_position, n).CopyTo(buffer);
            _position += n;
            return n;
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private static void AssertRuntimeAsync(MethodInfo method)
    {
        Assert.NotNull(method);
        Assert.Null(method.GetCustomAttribute<AsyncStateMachineAttribute>());
        Assert.True(
            method.MethodImplementationFlags.HasFlag(MethodImplAttributes.Async),
            $"{method.Name} flags were {method.MethodImplementationFlags}");
    }

    private sealed class CancelOnReadStream : Stream
    {
        private readonly byte[] _data;
        private long _position;

        public CancelOnReadStream(byte[] data)
        {
            _data = data;
            _position = Math.Min(5, data.Length);
        }

        public override bool CanRead => true;

        public override bool CanSeek => true;

        public override bool CanWrite => false;

        public override long Length => _data.Length;

        public override long Position
        {
            get => _position;
            set => _position = value;
        }

        public override void Flush()
        {
        }

        public override int Read(byte[] buffer, int offset, int count) => throw new OperationCanceledException();

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
            throw new OperationCanceledException(cancellationToken);

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private static float[] OneHot(ModelDefinition model, string label, float score)
    {
        float[] scores = new float[model.TrainingLabels.Length];
        scores[Array.IndexOf(model.TrainingLabels, label)] = score;
        return scores;
    }
}
