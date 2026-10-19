using System.Buffers;
using System.Runtime.ExceptionServices;
using System.Threading.Channels;
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
    /// Files are read in windows of <see cref="StreamWindowSize"/> so a window can be scored while the next one is read.
    /// Model inputs are scored in batches on the inference pool.
    /// Directories, symlinks, empty files, and short inputs use the same rules as <see cref="IdentifyPath"/>.
    /// The first read error, in input order, is thrown. Later windows are not read.
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

        var channel = OpenWindows();
        Exception? producerError = null;
        Task producer = Task.Run(() =>
        {
            try
            {
                Produce(paths, channel.Writer);
            }
            catch (ChannelClosedException)
            {
            }
            catch (Exception ex)
            {
                producerError = ex;
                channel.Writer.TryComplete(ex);
            }
        });

        Exception? consumeError = null;
        try
        {
            Consume(channel.Reader, paths, results);
        }
        catch (Exception ex)
        {
            consumeError = ex;
        }
        finally
        {
            channel.Writer.TryComplete();
        }

        try
        {
            producer.GetAwaiter().GetResult();
        }
        catch (ChannelClosedException)
        {
        }
        catch (Exception ex) when (producerError is null)
        {
            producerError = ex;
        }

        if (consumeError is not null)
        {
            ExceptionDispatchInfo.Capture(consumeError).Throw();
        }

        if (producerError is not null)
        {
            ExceptionDispatchInfo.Capture(producerError).Throw();
        }

        return results;
    }

    /// <summary>
    /// Identifies many filesystem paths without blocking the caller on IO.
    /// Paths are read in windows of <see cref="StreamWindowSize"/> and scored as each window arrives.
    /// Cancellation is observed between reads. A forward pass already running finishes.
    /// Results match <paramref name="paths"/> order.
    /// The first read error, in input order, is thrown. Later windows are not read.
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

        var channel = OpenWindows();
        Task producing = ProduceAsync(paths, channel.Writer, cancellationToken);
        try
        {
            await ConsumeAsync(channel.Reader, paths, results, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            channel.Writer.TryComplete();
            try
            {
                await producing.ConfigureAwait(false);
            }
            catch (ChannelClosedException)
            {
            }
        }

        return results;
    }

    private const int StreamWindowSize = 64;

    private static Channel<StreamWindow> OpenWindows() =>
        Channel.CreateBounded<StreamWindow>(new BoundedChannelOptions(2)
        {
            SingleReader = true,
            SingleWriter = true,
            FullMode = BoundedChannelFullMode.Wait,
        });

    private void Produce(IReadOnlyList<string> paths, ChannelWriter<StreamWindow> writer)
    {
        int count = paths.Count;
        for (int start = 0; start < count; start += StreamWindowSize)
        {
            int take = Math.Min(StreamWindowSize, count - start);
            StreamWindow window = ReadWindow(paths, start, take);
            try
            {
                if (!writer.TryWrite(window))
                {
                    writer.WriteAsync(window).AsTask().ConfigureAwait(false).GetAwaiter().GetResult();
                }
            }
            catch (ChannelClosedException)
            {
                ArrayPool<ushort>.Shared.Return(window.Tokens);
                return;
            }
        }

        writer.TryComplete();
    }

    private async Task ProduceAsync(IReadOnlyList<string> paths, ChannelWriter<StreamWindow> writer, CancellationToken cancellationToken)
    {
        try
        {
            int count = paths.Count;
            for (int start = 0; start < count; start += StreamWindowSize)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int take = Math.Min(StreamWindowSize, count - start);
                StreamWindow window = await ReadWindowAsync(paths, start, take, cancellationToken).ConfigureAwait(false);
                try
                {
                    await writer.WriteAsync(window, cancellationToken).ConfigureAwait(false);
                }
                catch (ChannelClosedException)
                {
                    ArrayPool<ushort>.Shared.Return(window.Tokens);
                    return;
                }
            }

            writer.TryComplete();
        }
        catch (ChannelClosedException)
        {
        }
        catch (Exception ex)
        {
            writer.TryComplete(ex);
            throw;
        }
    }

    private void Consume(ChannelReader<StreamWindow> reader, IReadOnlyList<string> paths, MagikaResult[] results)
    {
        try
        {
            while (reader.WaitToReadAsync().AsTask().ConfigureAwait(false).GetAwaiter().GetResult())
            {
                while (reader.TryRead(out StreamWindow? window))
                {
                    try
                    {
                        PublishWindow(window, paths, results);
                    }
                    finally
                    {
                        ArrayPool<ushort>.Shared.Return(window.Tokens);
                    }
                }
            }
        }
        catch
        {
            while (reader.TryRead(out StreamWindow? extra))
            {
                ArrayPool<ushort>.Shared.Return(extra.Tokens);
            }

            throw;
        }
    }

    private async Task ConsumeAsync(ChannelReader<StreamWindow> reader, IReadOnlyList<string> paths, MagikaResult[] results, CancellationToken cancellationToken)
    {
        try
        {
            while (await reader.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
            {
                while (reader.TryRead(out StreamWindow? window))
                {
                    try
                    {
                        PublishWindow(window, paths, results);
                    }
                    finally
                    {
                        ArrayPool<ushort>.Shared.Return(window.Tokens);
                    }
                }
            }
        }
        catch
        {
            while (reader.TryRead(out StreamWindow? extra))
            {
                ArrayPool<ushort>.Shared.Return(extra.Tokens);
            }

            throw;
        }
    }

    private StreamWindow ReadWindow(IReadOnlyList<string> paths, int start, int take)
    {
        int width = _model.FeaturesSize;
        ushort[] tokens = ArrayPool<ushort>.Shared.Rent(take * width);
        var prepared = new Prepared[take];
        try
        {
            Parallel.For(
                0,
                take,
                new ParallelOptions { MaxDegreeOfParallelism = IoParallelism(take) },
                i => prepared[i] = PreparePath(paths[start + i], tokens.AsSpan(i * width, width)));
            return new StreamWindow(start, take, tokens, prepared);
        }
        catch
        {
            ArrayPool<ushort>.Shared.Return(tokens);
            throw;
        }
    }

    private async Task<StreamWindow> ReadWindowAsync(IReadOnlyList<string> paths, int start, int take, CancellationToken cancellationToken)
    {
        int width = _model.FeaturesSize;
        ushort[] tokens = ArrayPool<ushort>.Shared.Rent(take * width);
        var prepared = new Prepared[take];
        try
        {
            await Parallel.ForEachAsync(
                Enumerable.Range(0, take),
                new ParallelOptions
                {
                    CancellationToken = cancellationToken,
                    MaxDegreeOfParallelism = IoParallelism(take),
                },
                async (i, token) => prepared[i] = await PreparePathAsync(paths[start + i], tokens, i * width, token).ConfigureAwait(false)).ConfigureAwait(false);
            return new StreamWindow(start, take, tokens, prepared);
        }
        catch
        {
            ArrayPool<ushort>.Shared.Return(tokens);
            throw;
        }
    }

    private void PublishWindow(StreamWindow window, IReadOnlyList<string> paths, MagikaResult[] results)
    {
        Prepared[] prepared = window.Prepared;
        for (int i = 0; i < window.Count; i++)
        {
            if (prepared[i].Error is Exception error)
            {
                ExceptionDispatchInfo.Capture(error).Throw();
            }
        }

        int width = _model.FeaturesSize;
        int rows = 0;
        for (int i = 0; i < window.Count; i++)
        {
            if (prepared[i].Model)
            {
                rows++;
            }
            else
            {
                results[window.Start + i] = prepared[i].Early!;
            }
        }

        if (rows == 0)
        {
            return;
        }

        ushort[]? rented = null;
        try
        {
            ushort[] packed;
            int[] owner = new int[rows];
            if (rows == window.Count)
            {
                packed = window.Tokens;
                for (int i = 0; i < window.Count; i++)
                {
                    owner[i] = window.Start + i;
                }
            }
            else
            {
                rented = ArrayPool<ushort>.Shared.Rent(rows * width);
                packed = rented;
                int cursor = 0;
                for (int i = 0; i < window.Count; i++)
                {
                    if (!prepared[i].Model)
                    {
                        continue;
                    }

                    window.Tokens.AsSpan(i * width, width).CopyTo(packed.AsSpan(cursor * width, width));
                    owner[cursor++] = window.Start + i;
                }
            }

            InferPacked(packed, rows, owner, results, index => paths[index]);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<ushort>.Shared.Return(rented);
            }
        }
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

    private void InferPacked(ushort[] packed, int rows, int[] owner, MagikaResult[] results, Func<int, string?>? pathAt)
    {
        int width = _model.FeaturesSize;
        int batchSize = _options.BatchSize;
        int chunks = (rows + batchSize - 1) / batchSize;
        Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = _workerCount }, chunk =>
        {
            int start = chunk * batchSize;
            int take = Math.Min(batchSize, rows - start);
            int run = take;
            ushort[]? rented = null;
            ReadOnlySpan<ushort> tokens = packed.AsSpan(start * width, take * width);
            if (take < batchSize && start > 0)
            {
                run = batchSize;
                rented = ArrayPool<ushort>.Shared.Rent(run * width);
                tokens.CopyTo(rented);
                ReadOnlySpan<ushort> last = rented.AsSpan((take - 1) * width, width);
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
                    ArrayPool<ushort>.Shared.Return(rented);
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

                return lane.Session.Run(tokens, batch, sequence, _model.PaddingToken);
            }
            finally
            {
                Onnx.SimdGemm.ThreadBudget = previous;
            }
        }
    }

    private float[] RunLane(InferenceLane lane, ReadOnlySpan<ushort> tokens, int batch, int sequence)
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

                return lane.Session.RunCompact(tokens, batch, sequence, _model.PaddingToken);
            }
            finally
            {
                Onnx.SimdGemm.ThreadBudget = previous;
            }
        }
    }

    private Prepared PreparePath(string path, Span<ushort> features)
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

    private async Task<Prepared> PreparePathAsync(string path, ushort[] staging, int offset, CancellationToken cancellationToken)
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

    private Prepared FromEnds(string path, ReadOnlySpan<byte> prefix, ReadOnlySpan<byte> suffix, long length, Span<ushort> features)
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

    private sealed class StreamWindow(int start, int count, ushort[] tokens, Prepared[] prepared)
    {
        public int Start { get; } = start;

        public int Count { get; } = count;

        public ushort[] Tokens { get; } = tokens;

        public Prepared[] Prepared { get; } = prepared;
    }

    private sealed class InferenceLane(Onnx.OnnxSession session, Lock gate)
    {
        public Onnx.OnnxSession Session { get; } = session;

        public Lock Gate { get; } = gate;
    }
}
