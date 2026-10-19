using Microsoft.Win32.SafeHandles;

namespace Magika;

/// <summary>
/// Identifies file content types with the bundled Magika <c>standard_v3_3</c> model.
/// A session is safe to use from multiple threads.
/// </summary>
public sealed class MagikaSession
{
    private readonly ModelDefinition _model;
    private readonly Onnx.OnnxSession _session;
    private readonly MagikaOptions _options;
    private readonly Lock _gate = new();

    private MagikaSession(ModelDefinition model, Onnx.OnnxSession session, MagikaOptions options)
    {
        _model = model;
        _session = session;
        _options = options;
    }

    /// <summary>Bundled model name. Currently <c>standard_v3_3</c>.</summary>
    public string ModelName => ModelDefinition.Name;

    /// <summary>Bundled model major version.</summary>
    public int ModelMajorVersion => ModelDefinition.MajorVersion;

    /// <summary>Creates a session with the default high-confidence options and the embedded model.</summary>
    public static MagikaSession Create() => Create(MagikaOptions.Default);

    /// <summary>Creates a session with the embedded standard_v3_3 model.</summary>
    public static MagikaSession Create(MagikaOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ModelDefinition model = ModelDefinition.LoadDefault();
        using Stream stream = ModelDefinition.OpenModel();
        using MemoryStream copy = new();
        stream.CopyTo(copy);
        Onnx.OnnxSession session = Onnx.OnnxSession.Load(copy.GetBuffer().AsSpan(0, (int)copy.Length));
        return new MagikaSession(model, session, options);
    }

    /// <summary>Identifies a buffer.</summary>
    public MagikaResult Identify(ReadOnlySpan<byte> content) => Identify(content, path: null);

    /// <summary>
    /// Identifies a stream. A seekable stream is read at the start and end only, and its position is restored.
    /// A non-seekable stream is consumed to the end; only the first and last block are kept.
    /// </summary>
    public MagikaResult Identify(Stream stream, string? path = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead)
        {
            throw new ArgumentException("Stream must be readable.", nameof(stream));
        }

        if (!stream.CanSeek)
        {
            using ContentEnds.Window ends = ContentEnds.ReadForward(stream, _model.BlockSize, out long length);
            if (length == 0)
            {
                return Heuristic(path, "empty", score: 1f);
            }

            return IdentifyBlocks(ends.Prefix, ends.Suffix, length, path);
        }

        long position = stream.Position;
        try
        {
            long length = stream.Length;
            if (length == 0)
            {
                return Heuristic(path, "empty", score: 1f);
            }

            using ContentEnds.Window ends = ContentEnds.ReadSeekable(stream, length, _model.BlockSize);
            return IdentifyBlocks(ends.Prefix, ends.Suffix, length, path);
        }
        finally
        {
            stream.Position = position;
        }
    }

    /// <summary>Identifies a filesystem path. Directories and, when requested, symlinks are reported as such.</summary>
    public MagikaResult IdentifyPath(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        FileAttributes attributes;
        try
        {
            attributes = File.GetAttributes(path);
        }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException)
        {
            throw new FileNotFoundException($"Could not find '{path}'.", path, ex);
        }

        bool link = (attributes & FileAttributes.ReparsePoint) != 0;
        if (link && !_options.FollowSymlinks)
        {
            return Heuristic(path, "symlink", score: 1f);
        }

        if (Directory.Exists(path))
        {
            return Heuristic(path, "directory", score: 1f);
        }

        if (!File.Exists(path))
        {
            return Heuristic(path, "unsupported", score: 1f);
        }

        // pread of the head and tail. A mapped view was slower for the 4 KiB block.
        using SafeFileHandle handle = File.OpenHandle(path);
        long length = RandomAccess.GetLength(handle);
        if (length == 0)
        {
            return Heuristic(path, "empty", score: 1f);
        }

        using ContentEnds.Window ends = ContentEnds.ReadHandle(handle, length, _model.BlockSize);
        return IdentifyBlocks(ends.Prefix, ends.Suffix, length, path);
    }

    internal bool EmbeddingFused => _session.EmbeddingFused;

    internal ModelDefinition Model => _model;

    internal float[] Score(ReadOnlySpan<byte> content)
    {
        int[] features = FeatureExtractor.Extract(content, _model);
        if (!FeatureExtractor.HasEnoughContent(features, _model))
        {
            throw new InvalidOperationException("Content is too small for the model.");
        }

        lock (_gate)
        {
            return _session.Run(features, batch: 1, _model.FeaturesSize);
        }
    }

    internal MagikaResult DecideScores(float[] scores, string? path = null)
    {
        ModelDecision decision = PostProcessor.Decide(_model, _options.PredictionMode, scores);
        return new MagikaResult(
            path,
            ContentTypes.Get(decision.Label),
            ContentTypes.Get(decision.TrainingLabel),
            decision.Score,
            decision.Reason);
    }

    private MagikaResult Identify(ReadOnlySpan<byte> content, string? path)
    {
        if (content.Length == 0)
        {
            return Heuristic(path, "empty", score: 1f);
        }

        int block = Math.Min(_model.BlockSize, content.Length);
        return IdentifyBlocks(content[..block], content[^block..], content.Length, path);
    }

    private MagikaResult IdentifyBlocks(ReadOnlySpan<byte> prefix, ReadOnlySpan<byte> suffix, long length, string? path)
    {
        if (!_options.UseModel)
        {
            return Heuristic(path, FeatureExtractor.ClassifySmall(prefix, length), score: 1f);
        }

        int[] features = FeatureExtractor.ExtractBlocks(prefix, suffix, _model);
        if (!FeatureExtractor.HasEnoughContent(features, _model))
        {
            return Heuristic(path, FeatureExtractor.ClassifySmall(prefix, length), score: 1f);
        }

        float[] scores;
        lock (_gate)
        {
            scores = _session.Run(features, batch: 1, features.Length);
        }

        return DecideScores(scores, path);
    }

    private static MagikaResult Heuristic(string? path, string label, float score) =>
        new(path, ContentTypes.Get(label), modelContentType: null, score, overwriteReason: null);
}
