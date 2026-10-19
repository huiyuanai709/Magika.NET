using System.Buffers;
using System.Runtime.ExceptionServices;
using Microsoft.Win32.SafeHandles;

namespace Magika;

public sealed partial class MagikaSession
{
    /// <summary>
    /// Scores one feature vector of the model feature size (2048 for <c>standard_v3_3</c>).
    /// The row must contain enough non-padding tokens for the model to run.
    /// </summary>
    public MagikaResult IdentifyFeatures(ReadOnlySpan<int> features, string? path = null)
    {
        string?[]? paths = path is null ? null : [path];
        return IdentifyFeaturesBatch(features, batch: 1, paths)[0];
    }

    /// <summary>
    /// Scores <paramref name="batch"/> feature vectors packed back-to-back.
    /// <paramref name="features"/> length must be <paramref name="batch"/> times the model feature size.
    /// Rows run in one forward pass, or a few passes when the batch is larger than <see cref="MagikaOptions.BatchSize"/>.
    /// Results follow the packed row order.
    /// </summary>
    public MagikaResult[] IdentifyFeaturesBatch(ReadOnlySpan<int> features, int batch, IReadOnlyList<string?>? paths = null)
    {
        int width = _model.FeaturesSize;
        if (batch <= 0 || features.Length != batch * width)
        {
            throw new ArgumentException("Feature buffer does not match the batch and the model feature size.", nameof(features));
        }

        if (paths is not null && paths.Count < batch)
        {
            throw new ArgumentException("Path list is shorter than the batch.", nameof(paths));
        }

        for (int i = 0; i < batch; i++)
        {
            if (!FeatureExtractor.HasEnoughContent(features.Slice(i * width, width), _model))
            {
                throw new ArgumentException($"Feature row {i} is too short for the model.", nameof(features));
            }
        }

        int[] packed = new int[features.Length];
        features.CopyTo(packed);
        int[] owner = new int[batch];
        for (int i = 0; i < batch; i++)
        {
            owner[i] = i;
        }

        MagikaResult[] results = new MagikaResult[batch];
        InferPacked(packed, batch, owner, results, paths is null ? null : index => paths[index]);
        return results;
    }

    /// <summary>
    /// Identifies many filesystem paths. Results match <paramref name="paths"/> order.
    /// Files are read concurrently. Model inputs are scored in batches on the inference pool.
    /// Directories, symlinks, empty files, and short inputs use the same rules as <see cref="IdentifyPath"/>.
    /// The first read error, in input order, is thrown after the reads finish.
    /// </summary>
    public MagikaResult[] IdentifyPaths(IReadOnlyList<string> paths)
    {
        ArgumentNullException.ThrowIfNull(paths);
        int count = paths.Count;
        MagikaResult[] results = new MagikaResult[count];
        if (count == 0)
        {
            return results;
        }

        int width = _model.FeaturesSize;
        int[] staging = new int[count * width];
        Prepared[] prepared = new Prepared[count];
        Parallel.For(
            0,
            count,
            new ParallelOptions { MaxDegreeOfParallelism = IoParallelism(count) },
            i => prepared[i] = PreparePath(paths[i], staging.AsSpan(i * width, width)));
        Publish(paths, staging, prepared, results);
        return results;
    }

    /// <summary>
    /// Identifies many filesystem paths without blocking the caller on IO.
    /// Cancellation is observed between reads. A forward pass already running finishes.
    /// Results match <paramref name="paths"/> order.
    /// </summary>
    public async Task<MagikaResult[]> IdentifyPathsAsync(IReadOnlyList<string> paths, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(paths);
        cancellationToken.ThrowIfCancellationRequested();
        int count = paths.Count;
        MagikaResult[] results = new MagikaResult[count];
        if (count == 0)
        {
            return results;
        }

        int width = _model.FeaturesSize;
        int[] staging = new int[count * width];
        Prepared[] prepared = new Prepared[count];
        await Parallel.ForEachAsync(
            Enumerable.Range(0, count),
            new ParallelOptions
            {
                CancellationToken = cancellationToken,
                MaxDegreeOfParallelism = IoParallelism(count),
            },
            async (i, token) => prepared[i] = await PreparePathAsync(paths[i], staging, i * width, token).ConfigureAwait(false)).ConfigureAwait(false);
        Publish(paths, staging, prepared, results);
        return results;
    }

    private void Publish(IReadOnlyList<string> paths, int[] staging, Prepared[] prepared, MagikaResult[] results)
    {
        for (int i = 0; i < prepared.Length; i++)
        {
            if (prepared[i].Error is Exception error)
            {
                ExceptionDispatchInfo.Capture(error).Throw();
            }
        }

        int width = _model.FeaturesSize;
        int rows = 0;
        for (int i = 0; i < prepared.Length; i++)
        {
            if (prepared[i].Model)
            {
                rows++;
            }
            else
            {
                results[i] = prepared[i].Early!;
            }
        }

        if (rows == 0)
        {
            return;
        }

        int[] packed;
        int[] owner = new int[rows];
        if (rows == prepared.Length)
        {
            packed = staging;
            for (int i = 0; i < prepared.Length; i++)
            {
                owner[i] = i;
            }
        }
        else
        {
            packed = new int[rows * width];
            int cursor = 0;
            for (int i = 0; i < prepared.Length; i++)
            {
                if (!prepared[i].Model)
                {
                    continue;
                }

                staging.AsSpan(i * width, width).CopyTo(packed.AsSpan(cursor * width, width));
                owner[cursor++] = i;
            }
        }

        InferPacked(packed, rows, owner, results, index => paths[index]);
    }

    private void InferPacked(int[] packed, int rows, int[] owner, MagikaResult[] results, Func<int, string?>? pathAt)
    {
        int width = _model.FeaturesSize;
        int batchSize = _options.BatchSize;
        int chunks = (rows + batchSize - 1) / batchSize;
        Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = _workerCount }, chunk =>
        {
            int start = chunk * batchSize;
            int take = Math.Min(batchSize, rows - start);
            int run = take;
            int[]? rented = null;
            ReadOnlySpan<int> tokens = packed.AsSpan(start * width, take * width);
            if (take < batchSize && start > 0)
            {
                run = batchSize;
                rented = ArrayPool<int>.Shared.Rent(run * width);
                tokens.CopyTo(rented);
                ReadOnlySpan<int> last = rented.AsSpan((take - 1) * width, width);
                for (int pad = take; pad < run; pad++)
                {
                    last.CopyTo(rented.AsSpan(pad * width, width));
                }

                tokens = rented.AsSpan(0, run * width);
            }

            try
            {
                InferenceLane[] lanes = Pool();
            InferenceLane lane = lanes[chunk % lanes.Length];
                float[] scores = RunLane(lane, tokens, run, width);
                int labels = scores.Length / run;
                for (int row = 0; row < take; row++)
                {
                    int index = owner[start + row];
                    results[index] = DecideScores(scores.AsSpan(row * labels, labels), pathAt?.Invoke(index));
                }
            }
            finally
            {
                if (rented is not null)
                {
                    ArrayPool<int>.Shared.Return(rented);
                }
            }
        });
    }

    private float[] RunLane(InferenceLane lane, ReadOnlySpan<int> tokens, int batch, int sequence)
    {
        lock (lane.Gate)
        {
            int previous = Onnx.SimdGemm.ThreadBudget;
            try
            {
                if (_gemmThreadsPerLane > 0)
                {
                    Onnx.SimdGemm.ThreadBudget = _gemmThreadsPerLane;
                }

                return lane.Session.Run(tokens, batch, sequence);
            }
            finally
            {
                Onnx.SimdGemm.ThreadBudget = previous;
            }
        }
    }

    private Prepared PreparePath(string path, Span<int> features)
    {
        try
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            if (ClassifySpecialPath(path) is MagikaResult special)
            {
                return Prepared.Done(special);
            }

            using SafeFileHandle handle = File.OpenHandle(path);
            long length = RandomAccess.GetLength(handle);
            if (length == 0)
            {
                return Prepared.Done(Heuristic(path, "empty", score: 1f));
            }

            using ContentEnds.Window ends = ContentEnds.ReadHandle(handle, length, _model.BlockSize);
            return FromEnds(path, ends.Prefix, ends.Suffix, length, features);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FileNotFoundException or ArgumentException)
        {
            return Prepared.Failed(ex);
        }
    }

    private async Task<Prepared> PreparePathAsync(string path, int[] staging, int offset, CancellationToken cancellationToken)
    {
        try
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            cancellationToken.ThrowIfCancellationRequested();
            if (ClassifySpecialPath(path) is MagikaResult special)
            {
                return Prepared.Done(special);
            }

            using SafeFileHandle handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.Asynchronous);
            long length = RandomAccess.GetLength(handle);
            if (length == 0)
            {
                return Prepared.Done(Heuristic(path, "empty", score: 1f));
            }

            using ContentEnds.Lease ends = await ContentEnds.ReadHandleAsync(handle, length, _model.BlockSize, cancellationToken).ConfigureAwait(false);
            return FromEnds(path, ends.Prefix, ends.Suffix, length, staging.AsSpan(offset, _model.FeaturesSize));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FileNotFoundException or ArgumentException)
        {
            return Prepared.Failed(ex);
        }
    }

    private Prepared FromEnds(string path, ReadOnlySpan<byte> prefix, ReadOnlySpan<byte> suffix, long length, Span<int> features)
    {
        if (!_options.UseModel)
        {
            return Prepared.Done(Heuristic(path, FeatureExtractor.ClassifySmall(prefix, length), score: 1f));
        }

        FeatureExtractor.ExtractBlocks(prefix, suffix, _model, features);
        if (!FeatureExtractor.HasEnoughContent(features, _model))
        {
            return Prepared.Done(Heuristic(path, FeatureExtractor.ClassifySmall(prefix, length), score: 1f));
        }

        return Prepared.ForModel();
    }

    private static int ResolveWorkers(int configured, bool specialized)
    {
        if (configured > 0)
        {
            return configured;
        }

        // The specialized convolution already fans across processors inside one lane.
        // One lane per core then runs that many full products at once and loses to a single wide product
        // (measured: 8 lanes at ~43 files/s vs 1 lane at ~75 files/s on an 8-core host).
        // The interpreter has no such inner parallel product, so it still wants one narrow lane per core.
        if (specialized)
        {
            return 1;
        }

        return Math.Max(1, Environment.ProcessorCount);
    }

    private InferenceLane[] Pool()
    {
        InferenceLane[]? pool = Volatile.Read(ref _pool);
        if (pool is not null)
        {
            return pool;
        }

        lock (_gate)
        {
            if (_pool is not null)
            {
                return _pool;
            }

            InferenceLane[] lanes = new InferenceLane[_workerCount];
            lanes[0] = new InferenceLane(_session, _gate);
            for (int i = 1; i < lanes.Length; i++)
            {
                lanes[i] = new InferenceLane(_session.Fork(), new Lock());
            }

            Volatile.Write(ref _pool, lanes);
            return lanes;
        }
    }

    private static int IoParallelism(int count) => Math.Min(count, Math.Max(4, Environment.ProcessorCount));

    private readonly record struct Prepared(bool Model, MagikaResult? Early, Exception? Error)
    {
        public static Prepared Done(MagikaResult result) => new(false, result, null);

        public static Prepared ForModel() => new(true, null, null);

        public static Prepared Failed(Exception error) => new(false, null, error);
    }

    private sealed class InferenceLane(Onnx.OnnxSession session, Lock gate)
    {
        public Onnx.OnnxSession Session { get; } = session;

        public Lock Gate { get; } = gate;
    }
}
