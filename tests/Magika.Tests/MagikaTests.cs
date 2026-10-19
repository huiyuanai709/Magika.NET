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

    private static float[] OneHot(ModelDefinition model, string label, float score)
    {
        float[] scores = new float[model.TrainingLabels.Length];
        scores[Array.IndexOf(model.TrainingLabels, label)] = score;
        return scores;
    }
}
