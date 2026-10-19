using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Magika.Onnx;

/// <summary>
/// Per-lane activation state for <see cref="StandardV33Model"/>. Not thread-safe.
/// One evaluation at a time, matching <see cref="OnnxSession"/>.
/// Convolution reads <c>_conv</c> in place: each channel is a row of <see cref="StandardV33Model.RowStride"/>
/// groups, so the GEMM never materializes an im2col panel.
/// </summary>
internal sealed class StandardV33Session
{
    private const int ChannelBlock = 32;

    private readonly StandardV33Model _model;
    private readonly Action<int> _convolve;
    private readonly float[] _lnBlock = new float[16 * StandardV33Model.Channels];
    private readonly float[] _sumSq = new float[StandardV33Model.Channels];
    private readonly int[] _tileOh = new int[StandardV33Model.OutH];
    private readonly int[] _tileLo = new int[StandardV33Model.OutH];
    private readonly int[] _tileHi = new int[StandardV33Model.OutH];
    private int _batch;
    private int _threads;
    private int _capacity;
    private float[] _conv = [];
    private float[] _tileRaw = [];
    private float[] _pooled = [];
    private float[] _norm = new float[StandardV33Model.Filters];
    private float[] _logits = new float[StandardV33Model.Classes];
    private float[] _portAcc = [];
    private readonly float[] _mean = new float[StandardV33Model.Channels];
    private readonly float[] _invStd = new float[StandardV33Model.Channels];
    private float[] _padT = [];
    private readonly float[] _padAcc = new float[StandardV33Model.Filters];
    private byte[] _shortcut = [];
    private float[] _savedMean = [];
    private float[] _savedInv = [];
    private float[] _analytic = [];
    private ConvShape _shape;
    private int _geomTiles;
    private int _geomSample;

    internal static void CheckKernels()
    {
        if (StandardV33Model.RowStride < StandardV33Model.SampleStride + StandardV33Model.Kernel - 1)
        {
            throw new InvalidOperationException("Convolution row stride is shorter than a trailing vector load.");
        }

        CheckTranspose();
        if (Avx512F.IsSupported)
        {
            CheckDirect(avx512: true);
        }

        if (Avx2.IsSupported && Fma.IsSupported)
        {
            CheckDirect(avx512: false);
        }
    }

    private static unsafe void CheckTranspose()
    {
        const int groups = 16;
        const int channels = StandardV33Model.Channels;
        float[] block = new float[groups * channels];
        float[] conv = new float[channels * StandardV33Model.RowStride];
        for (int g = 0; g < groups; g++)
        {
            for (int c = 0; c < channels; c++)
            {
                block[(g * channels) + c] = (g * 10000) + c + 0.5f;
            }
        }

        fixed (float* source = block)
        fixed (float* dest = conv)
        {
            for (int c0 = 0; c0 < channels; c0 += 16)
            {
                Transpose16(source, channels, c0, dest, StandardV33Model.RowStride, 0);
            }
        }

        for (int c = 0; c < channels; c++)
        {
            for (int g = 0; g < groups; g++)
            {
                float actual = conv[(c * StandardV33Model.RowStride) + g];
                float expected = (g * 10000) + c + 0.5f;
                if (actual != expected)
                {
                    throw new InvalidOperationException($"Transpose mismatch channel {c} group {g} actual {actual} expected {expected}.");
                }
            }
        }
    }

    private static unsafe void CheckDirect(bool avx512)
    {
        const int channels = 4;
        const int k = channels * StandardV33Model.Kernel;
        const int stride = StandardV33Model.RowStride;
        const int oh0 = 4;
        int nr = avx512 ? 32 : 16;
        float[] a = new float[k * StandardV33Model.Mr];
        float[] conv = new float[channels * stride];
        float[] c = new float[StandardV33Model.Mr * nr];
        var random = new Random(avx512 ? 5 : 9);
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)((random.NextDouble() * 2) - 1);
        }

        for (int i = 0; i < conv.Length; i++)
        {
            conv[i] = (float)((random.NextDouble() * 2) - 1);
        }

        fixed (float* ap = a)
        fixed (float* bp = conv)
        fixed (float* cp = c)
        {
            if (avx512)
            {
                Kernel512_8x32(ap, bp + oh0, stride, channels, cp, nr, accumulate: false);
            }
            else
            {
                Kernel256_8x16(ap, bp + oh0, stride, channels, cp, nr, accumulate: false);
            }
        }

        for (int row = 0; row < StandardV33Model.Mr; row++)
        {
            for (int col = 0; col < nr; col++)
            {
                float sum = 0f;
                for (int ic = 0; ic < channels; ic++)
                {
                    for (int kh = 0; kh < StandardV33Model.Kernel; kh++)
                    {
                        int depth = (ic * StandardV33Model.Kernel) + kh;
                        sum += a[(depth * StandardV33Model.Mr) + row] * conv[(ic * stride) + oh0 + kh + col];
                    }
                }

                float actual = c[(row * nr) + col];
                float delta = MathF.Abs(sum - actual);
                if (delta > 1e-3f)
                {
                    throw new InvalidOperationException($"Direct kernel mismatch row {row} col {col} delta {delta}.");
                }
            }
        }
    }

    internal StandardV33Session(StandardV33Model model)
    {
        _model = model;
        _convolve = ConvolveWorker;
        Ensure(1);
    }

    internal unsafe float[] Evaluate(ReadOnlySpan<int> tokens, int batch, int paddingToken = -1)
    {
        if (batch <= 0 || tokens.Length != batch * StandardV33Model.Sequence)
        {
            throw new ArgumentException("Token buffer does not match the standard_v3_3 sequence.", nameof(tokens));
        }

        fixed (int* pointer = tokens)
        {
            return EvaluateCore(new IntTokens(pointer), batch, paddingToken);
        }
    }

    internal unsafe float[] Evaluate(ReadOnlySpan<ushort> tokens, int batch, int paddingToken = -1)
    {
        if (batch <= 0 || tokens.Length != batch * StandardV33Model.Sequence)
        {
            throw new ArgumentException("Token buffer does not match the standard_v3_3 sequence.", nameof(tokens));
        }

        fixed (ushort* pointer = tokens)
        {
            return EvaluateCore(new UShortTokens(pointer), batch, paddingToken);
        }
    }

    private unsafe float[] EvaluateCore<T>(T tokens, int batch, int paddingToken) where T : struct, ITokenReader
    {
        Ensure(batch);
        _batch = batch;
        bool shortcut = false;
        fixed (float* convRaw = _conv)
        {
            float* conv = Align64(convRaw);
            int sampleStride = StandardV33Model.Channels * StandardV33Model.RowStride;
            for (int sample = 0; sample < batch; sample++)
            {
                int origin = sample * StandardV33Model.Sequence;
                EmbedAndNormalize(tokens, origin, conv + (sample * sampleStride));
                bool sparse = IsPadShortcut(tokens, origin, paddingToken);
                _shortcut[sample] = sparse ? (byte)1 : (byte)0;
                if (sparse)
                {
                    shortcut = true;
                    _mean.AsSpan().CopyTo(_savedMean.AsSpan(sample * StandardV33Model.Channels, StandardV33Model.Channels));
                    _invStd.AsSpan().CopyTo(_savedInv.AsSpan(sample * StandardV33Model.Channels, StandardV33Model.Channels));
                }
            }
        }

        if (!shortcut)
        {
            if (_model.Portable)
            {
                for (int sample = 0; sample < batch; sample++)
                {
                    ConvolvePortable(sample, ReadOnlySpan<int>.Empty, ReadOnlySpan<int>.Empty);
                }
            }
            else
            {
                _pooled.AsSpan(0, batch * StandardV33Model.Filters).Fill(float.NegativeInfinity);
                _shape = ConvShape.Dense;
                _geomTiles = batch * _model.TilesPerSample;
                RunGeometry();
            }
        }
        else
        {
            ConvolveShortcutBatch(tokens);
        }

        // GELU is monotonic, so max(gelu(conv + bias)) == gelu(max(conv) + bias).
        // The bias is one value per filter. This drops the spatial tanh down to 512 values.
        ActivatePooled();

        float[] owned = GC.AllocateUninitializedArray<float>(batch * StandardV33Model.Classes);
        for (int sample = 0; sample < batch; sample++)
        {
            Project(sample, owned.AsSpan(sample * StandardV33Model.Classes, StandardV33Model.Classes));
        }

        return owned;
    }

    private void Ensure(int batch)
    {
        if (batch <= _capacity)
        {
            return;
        }

        _conv = GC.AllocateUninitializedArray<float>((batch * StandardV33Model.Channels * StandardV33Model.RowStride) + 16);
        _pooled = GC.AllocateUninitializedArray<float>(batch * StandardV33Model.Filters);
        _shortcut = new byte[batch];
        _savedMean = new float[batch * StandardV33Model.Channels];
        _savedInv = new float[batch * StandardV33Model.Channels];
        _analytic = new float[batch * StandardV33Model.Filters];
        _padT = new float[batch * StandardV33Model.Kernel * StandardV33Model.Filters];
        if (!_model.Portable)
        {
            _tileRaw = GC.AllocateUninitializedArray<float>((StandardV33Model.Filters * _model.Nr) + 16);
        }
        else
        {
            _portAcc = new float[StandardV33Model.OutH];
        }

        _capacity = batch;
    }

    private unsafe void EmbedAndNormalize<T>(T tokens, int origin, float* conv) where T : struct, ITokenReader
    {
        int groups = StandardV33Model.Groups;
        int channels = StandardV33Model.Channels;
        _mean.AsSpan().Clear();
        _sumSq.AsSpan().Clear();
        fixed (float* table = _model.Embed)
        fixed (float* sum = _mean)
        fixed (float* sumSq = _sumSq)
        fixed (float* invStd = _invStd)
        {
            for (int g = 0; g < groups; g++)
            {
                int token = origin + (g * StandardV33Model.TokensPerGroup);
                for (int t = 0; t < StandardV33Model.TokensPerGroup; t++)
                {
                    int id = tokens.At(token + t);
                    float* src = Embedding(table, id);
                    AccumulateToken(src, sum + (t * StandardV33Model.EmbedDim), sumSq + (t * StandardV33Model.EmbedDim));
                }
            }

            FinalizeStats(sum, sumSq, invStd, _model.InvLength, _model.Epsilon);
            fixed (float* block = _lnBlock)
            fixed (float* gamma = _model.Ln0Gamma)
            fixed (float* beta = _model.Ln0Beta)
            {
                for (int g0 = 0; g0 < groups; g0 += 16)
                {
                    for (int i = 0; i < 16; i++)
                    {
                        int g = g0 + i;
                        float* row = block + (i * channels);
                        int token = origin + (g * StandardV33Model.TokensPerGroup);
                        for (int t = 0; t < StandardV33Model.TokensPerGroup; t++)
                        {
                            int id = tokens.At(token + t);
                            CopyToken(Embedding(table, id), row + (t * StandardV33Model.EmbedDim));
                        }

                        NormalizeRow(row, sum, invStd, gamma[g], beta[g]);
                    }

                    for (int c0 = 0; c0 < channels; c0 += 16)
                    {
                        Transpose16(block, channels, c0, conv, StandardV33Model.RowStride, g0);
                    }
                }
            }

            ZeroTails(conv);
        }
    }

    private static unsafe float* Embedding(float* table, int id)
    {
        if ((uint)id >= (uint)StandardV33Model.Vocab)
        {
            throw new InvalidDataException($"Embedding index {id} is outside 0..{StandardV33Model.Vocab - 1}.");
        }

        return table + (id * StandardV33Model.EmbedDim);
    }

    private static unsafe void AccumulateToken(float* src, float* sum, float* sumSq)
    {
        int dim = StandardV33Model.EmbedDim;
        if (Avx512F.IsSupported)
        {
            for (int d = 0; d < dim; d += 16)
            {
                Vector512<float> value = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(src + d));
                Vector512<float> total = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(sum + d));
                Vector512<float> squares = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(sumSq + d));
                total = Avx512F.Add(total, value);
                squares = Avx512F.Add(squares, Avx512F.Multiply(value, value));
                total.StoreUnsafe(ref Unsafe.AsRef<float>(sum + d));
                squares.StoreUnsafe(ref Unsafe.AsRef<float>(sumSq + d));
            }

            return;
        }

        if (Avx2.IsSupported)
        {
            for (int d = 0; d < dim; d += 8)
            {
                Vector256<float> value = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(src + d));
                Vector256<float> total = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(sum + d));
                Vector256<float> squares = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(sumSq + d));
                total = Avx.Add(total, value);
                squares = Avx.Add(squares, Avx.Multiply(value, value));
                total.StoreUnsafe(ref Unsafe.AsRef<float>(sum + d));
                squares.StoreUnsafe(ref Unsafe.AsRef<float>(sumSq + d));
            }

            return;
        }

        for (int d = 0; d < dim; d++)
        {
            float value = src[d];
            sum[d] += value;
            sumSq[d] += value * value;
        }
    }

    /// <summary>
    /// Replaces <paramref name="sum"/> with the mean and writes the inverse std to <paramref name="invStd"/>.
    /// Group order matches the previous channel-major reduction, so the layer norm stays bit-identical on AVX.
    /// </summary>
    private static unsafe void FinalizeStats(float* sum, float* sumSq, float* invStd, float invN, float eps)
    {
        int channels = StandardV33Model.Channels;
        if (Avx512F.IsSupported)
        {
            Vector512<float> inv = Vector512.Create(invN);
            Vector512<float> epsilon = Vector512.Create(eps);
            for (int c = 0; c < channels; c += 16)
            {
                Vector512<float> total = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(sum + c));
                Vector512<float> squares = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(sumSq + c));
                Vector512<float> mean = Avx512F.Multiply(total, inv);
                Vector512<float> variance = Avx512F.Subtract(Avx512F.Multiply(squares, inv), Avx512F.Multiply(mean, mean));
                variance = Avx512F.Max(variance, Vector512<float>.Zero);
                Vector512<float> scale = Avx512F.Divide(Vector512<float>.One, Avx512F.Sqrt(Avx512F.Add(variance, epsilon)));
                mean.StoreUnsafe(ref Unsafe.AsRef<float>(sum + c));
                scale.StoreUnsafe(ref Unsafe.AsRef<float>(invStd + c));
            }

            return;
        }

        if (Avx2.IsSupported)
        {
            Vector256<float> inv = Vector256.Create(invN);
            Vector256<float> epsilon = Vector256.Create(eps);
            for (int c = 0; c < channels; c += 8)
            {
                Vector256<float> total = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(sum + c));
                Vector256<float> squares = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(sumSq + c));
                Vector256<float> mean = Avx.Multiply(total, inv);
                Vector256<float> variance = Avx.Subtract(Avx.Multiply(squares, inv), Avx.Multiply(mean, mean));
                variance = Avx.Max(variance, Vector256<float>.Zero);
                Vector256<float> scale = Avx.Divide(Vector256<float>.One, Avx.Sqrt(Avx.Add(variance, epsilon)));
                mean.StoreUnsafe(ref Unsafe.AsRef<float>(sum + c));
                scale.StoreUnsafe(ref Unsafe.AsRef<float>(invStd + c));
            }

            return;
        }

        for (int c = 0; c < channels; c++)
        {
            float mean = sum[c] * invN;
            float variance = MathF.Max((sumSq[c] * invN) - (mean * mean), 0f);
            sum[c] = mean;
            invStd[c] = 1f / MathF.Sqrt(variance + eps);
        }
    }

    private static unsafe void CopyToken(float* src, float* dst)
    {
        int dim = StandardV33Model.EmbedDim;
        if (Avx512F.IsSupported)
        {
            for (int d = 0; d < dim; d += 16)
            {
                Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(src + d)).StoreUnsafe(ref Unsafe.AsRef<float>(dst + d));
            }

            return;
        }

        if (Avx2.IsSupported)
        {
            for (int d = 0; d < dim; d += 8)
            {
                Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(src + d)).StoreUnsafe(ref Unsafe.AsRef<float>(dst + d));
            }

            return;
        }

        for (int d = 0; d < dim; d++)
        {
            dst[d] = src[d];
        }
    }

    private static unsafe void NormalizeRow(float* row, float* mean, float* invStd, float gamma, float beta)
    {
        int channels = StandardV33Model.Channels;
        if (Avx512F.IsSupported)
        {
            Vector512<float> gammaV = Vector512.Create(gamma);
            Vector512<float> betaV = Vector512.Create(beta);
            for (int c = 0; c < channels; c += 16)
            {
                Vector512<float> value = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(row + c));
                Vector512<float> scale = Avx512F.Multiply(Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(invStd + c)), gammaV);
                Vector512<float> centered = Avx512F.Subtract(value, Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(mean + c)));
                Avx512F.Add(Avx512F.Multiply(centered, scale), betaV).StoreUnsafe(ref Unsafe.AsRef<float>(row + c));
            }

            return;
        }

        if (Avx2.IsSupported)
        {
            Vector256<float> gammaV = Vector256.Create(gamma);
            Vector256<float> betaV = Vector256.Create(beta);
            for (int c = 0; c < channels; c += 8)
            {
                Vector256<float> value = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(row + c));
                Vector256<float> scale = Avx.Multiply(Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(invStd + c)), gammaV);
                Vector256<float> centered = Avx.Subtract(value, Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(mean + c)));
                Avx.Add(Avx.Multiply(centered, scale), betaV).StoreUnsafe(ref Unsafe.AsRef<float>(row + c));
            }

            return;
        }

        for (int c = 0; c < channels; c++)
        {
            row[c] = ((row[c] - mean[c]) * (invStd[c] * gamma)) + beta;
        }
    }

    private static unsafe void ZeroTails(float* conv)
    {
        int channels = StandardV33Model.Channels;
        int stride = StandardV33Model.RowStride;
        for (int ic = 0; ic < channels; ic++)
        {
            float* tail = conv + (ic * stride) + StandardV33Model.Groups;
            if (Avx512F.IsSupported)
            {
                Vector512<float>.Zero.StoreUnsafe(ref Unsafe.AsRef<float>(tail));
            }
            else if (Avx2.IsSupported)
            {
                Vector256<float>.Zero.StoreUnsafe(ref Unsafe.AsRef<float>(tail));
                Vector256<float>.Zero.StoreUnsafe(ref Unsafe.AsRef<float>(tail), 8);
            }
            else
            {
                for (int i = 0; i < stride - StandardV33Model.Groups; i++)
                {
                    tail[i] = 0f;
                }
            }
        }
    }

    /// <summary>
    /// 16-group tile. Each 16×16 quadrant is an 8×8 register transpose, then stored as contiguous groups.
    /// </summary>
    private static unsafe void Transpose16(float* block, int sourceStride, int c0, float* conv, int rowStride, int g0)
    {
        if (Avx2.IsSupported)
        {
            TransposeQuadrant(block, sourceStride, 0, c0, conv, rowStride, g0);
            TransposeQuadrant(block, sourceStride, 0, c0 + 8, conv, rowStride, g0);
            TransposeQuadrant(block, sourceStride, 8, c0, conv, rowStride, g0);
            TransposeQuadrant(block, sourceStride, 8, c0 + 8, conv, rowStride, g0);
            return;
        }

        for (int gi = 0; gi < 16; gi++)
        {
            for (int ci = 0; ci < 16; ci++)
            {
                conv[((c0 + ci) * rowStride) + g0 + gi] = block[(gi * sourceStride) + c0 + ci];
            }
        }
    }

    private static unsafe void TransposeQuadrant(float* block, int sourceStride, int rowBase, int channel0, float* conv, int rowStride, int g0)
    {
        Vector256<float> r0 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(block + ((rowBase + 0) * sourceStride) + channel0));
        Vector256<float> r1 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(block + ((rowBase + 1) * sourceStride) + channel0));
        Vector256<float> r2 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(block + ((rowBase + 2) * sourceStride) + channel0));
        Vector256<float> r3 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(block + ((rowBase + 3) * sourceStride) + channel0));
        Vector256<float> r4 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(block + ((rowBase + 4) * sourceStride) + channel0));
        Vector256<float> r5 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(block + ((rowBase + 5) * sourceStride) + channel0));
        Vector256<float> r6 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(block + ((rowBase + 6) * sourceStride) + channel0));
        Vector256<float> r7 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(block + ((rowBase + 7) * sourceStride) + channel0));
        Transpose8(r0, r1, r2, r3, r4, r5, r6, r7, out Vector256<float> c0, out Vector256<float> c1, out Vector256<float> c2, out Vector256<float> c3, out Vector256<float> c4, out Vector256<float> c5, out Vector256<float> c6, out Vector256<float> c7);
        int group = g0 + rowBase;
        Store8(conv + (channel0 * rowStride) + group, c0);
        Store8(conv + ((channel0 + 1) * rowStride) + group, c1);
        Store8(conv + ((channel0 + 2) * rowStride) + group, c2);
        Store8(conv + ((channel0 + 3) * rowStride) + group, c3);
        Store8(conv + ((channel0 + 4) * rowStride) + group, c4);
        Store8(conv + ((channel0 + 5) * rowStride) + group, c5);
        Store8(conv + ((channel0 + 6) * rowStride) + group, c6);
        Store8(conv + ((channel0 + 7) * rowStride) + group, c7);
    }

    private static unsafe void Store8(float* dest, Vector256<float> value) =>
        value.StoreUnsafe(ref Unsafe.AsRef<float>(dest));

    private static void Transpose8(
        Vector256<float> r0, Vector256<float> r1, Vector256<float> r2, Vector256<float> r3,
        Vector256<float> r4, Vector256<float> r5, Vector256<float> r6, Vector256<float> r7,
        out Vector256<float> c0, out Vector256<float> c1, out Vector256<float> c2, out Vector256<float> c3,
        out Vector256<float> c4, out Vector256<float> c5, out Vector256<float> c6, out Vector256<float> c7)
    {
        Vector256<float> t0 = Avx.UnpackLow(r0, r1);
        Vector256<float> t1 = Avx.UnpackHigh(r0, r1);
        Vector256<float> t2 = Avx.UnpackLow(r2, r3);
        Vector256<float> t3 = Avx.UnpackHigh(r2, r3);
        Vector256<float> t4 = Avx.UnpackLow(r4, r5);
        Vector256<float> t5 = Avx.UnpackHigh(r4, r5);
        Vector256<float> t6 = Avx.UnpackLow(r6, r7);
        Vector256<float> t7 = Avx.UnpackHigh(r6, r7);
        Vector256<float> u0 = Avx.UnpackLow(t0.AsDouble(), t2.AsDouble()).AsSingle();
        Vector256<float> u1 = Avx.UnpackHigh(t0.AsDouble(), t2.AsDouble()).AsSingle();
        Vector256<float> u2 = Avx.UnpackLow(t1.AsDouble(), t3.AsDouble()).AsSingle();
        Vector256<float> u3 = Avx.UnpackHigh(t1.AsDouble(), t3.AsDouble()).AsSingle();
        Vector256<float> u4 = Avx.UnpackLow(t4.AsDouble(), t6.AsDouble()).AsSingle();
        Vector256<float> u5 = Avx.UnpackHigh(t4.AsDouble(), t6.AsDouble()).AsSingle();
        Vector256<float> u6 = Avx.UnpackLow(t5.AsDouble(), t7.AsDouble()).AsSingle();
        Vector256<float> u7 = Avx.UnpackHigh(t5.AsDouble(), t7.AsDouble()).AsSingle();
        c0 = Avx.Permute2x128(u0, u4, 0x20);
        c1 = Avx.Permute2x128(u1, u5, 0x20);
        c2 = Avx.Permute2x128(u2, u6, 0x20);
        c3 = Avx.Permute2x128(u3, u7, 0x20);
        c4 = Avx.Permute2x128(u0, u4, 0x31);
        c5 = Avx.Permute2x128(u1, u5, 0x31);
        c6 = Avx.Permute2x128(u2, u6, 0x31);
        c7 = Avx.Permute2x128(u3, u7, 0x31);
    }

    /// <summary>
    /// Padding groups are four copies of one embedding. After layer norm their convolution
    /// factors into a per-channel scale, so a pure 5-tap window is five multiply-adds per filter
    /// instead of the full 1280-wide product. Content windows still use the GEMM.
    /// </summary>
    private void ConvolveShortcutBatch<T>(T tokens) where T : struct, ITokenReader
    {
        BuildPadResponses();
        Span<int> padStart = stackalloc int[StandardV33Model.OutH];
        Span<int> padEnd = stackalloc int[StandardV33Model.OutH];
        Span<int> activeStart = stackalloc int[StandardV33Model.OutH];
        Span<int> activeEnd = stackalloc int[StandardV33Model.OutH];
        for (int sample = 0; sample < _batch; sample++)
        {
            if (_shortcut[sample] == 0)
            {
                if (_model.Portable)
                {
                    ConvolvePortable(sample, ReadOnlySpan<int>.Empty, ReadOnlySpan<int>.Empty);
                }
                else
                {
                    _pooled.AsSpan(sample * StandardV33Model.Filters, StandardV33Model.Filters).Fill(float.NegativeInfinity);
                    _shape = ConvShape.Single;
                    _geomSample = sample;
                    _geomTiles = _model.TilesPerSample;
                    RunGeometry();
                }

                continue;
            }

            CollectPadWindows(tokens, sample * StandardV33Model.Sequence, padStart, padEnd, out int padRanges, activeStart, activeEnd, out int activeRanges, StandardV33Model.PadToken, out _);
            FillAnalytic(sample, padStart, padEnd, padRanges);
            if (activeRanges == 0)
            {
                _analytic.AsSpan(sample * StandardV33Model.Filters, StandardV33Model.Filters)
                    .CopyTo(_pooled.AsSpan(sample * StandardV33Model.Filters, StandardV33Model.Filters));
                continue;
            }

            if (_model.Portable)
            {
                ConvolvePortable(sample, activeStart[..activeRanges], activeEnd[..activeRanges]);
            }
            else
            {
                ConvolveRanges(sample, activeStart[..activeRanges], activeEnd[..activeRanges]);
            }

            CombineAnalytic(sample);
        }
    }

    private bool IsPadShortcut<T>(T tokens, int origin, int paddingToken) where T : struct, ITokenReader
    {
        if (paddingToken != StandardV33Model.PadToken)
        {
            return false;
        }

        // A pure window is five fully padded groups. Fewer pad tokens than that run cannot qualify.
        int need = (StandardV33Model.MinPureWindows + StandardV33Model.Kernel - 1) * StandardV33Model.TokensPerGroup;
        int pads = 0;
        for (int i = 0; i < StandardV33Model.Sequence; i++)
        {
            if (tokens.At(origin + i) == paddingToken)
            {
                pads++;
            }
        }

        if (pads < need)
        {
            return false;
        }

        Span<int> padStart = stackalloc int[StandardV33Model.OutH];
        Span<int> padEnd = stackalloc int[StandardV33Model.OutH];
        Span<int> activeStart = stackalloc int[StandardV33Model.OutH];
        Span<int> activeEnd = stackalloc int[StandardV33Model.OutH];
        CollectPadWindows(tokens, origin, padStart, padEnd, out _, activeStart, activeEnd, out _, paddingToken, out int pure);
        return pure >= StandardV33Model.MinPureWindows;
    }

    private void CollectPadWindows<T>(
        T tokens,
        int origin,
        Span<int> padStart,
        Span<int> padEnd,
        out int padRanges,
        Span<int> activeStart,
        Span<int> activeEnd,
        out int activeRanges,
        int paddingToken,
        out int pureWindows) where T : struct, ITokenReader
    {
        padRanges = 0;
        activeRanges = 0;
        pureWindows = 0;
        int groups = StandardV33Model.Groups;
        int kernel = StandardV33Model.Kernel;
        Span<byte> pure = stackalloc byte[groups];
        int step = StandardV33Model.TokensPerGroup;
        for (int g = 0; g < groups; g++)
        {
            int t = origin + (g * step);
            pure[g] = tokens.At(t) == paddingToken
                && tokens.At(t + 1) == paddingToken
                && tokens.At(t + 2) == paddingToken
                && tokens.At(t + 3) == paddingToken
                ? (byte)1
                : (byte)0;
        }

        int run = 0;
        for (int g = 0; g < kernel; g++)
        {
            run += pure[g];
        }

        int openPad = -1;
        int openActive = -1;
        for (int oh = 0; oh < StandardV33Model.OutH; oh++)
        {
            if (run == kernel)
            {
                pureWindows++;
                if (openActive >= 0)
                {
                    activeStart[activeRanges] = openActive;
                    activeEnd[activeRanges] = oh;
                    activeRanges++;
                    openActive = -1;
                }

                if (openPad < 0)
                {
                    openPad = oh;
                }
            }
            else
            {
                if (openPad >= 0)
                {
                    padStart[padRanges] = openPad;
                    padEnd[padRanges] = oh;
                    padRanges++;
                    openPad = -1;
                }

                if (openActive < 0)
                {
                    openActive = oh;
                }
            }

            if (oh + kernel < groups)
            {
                run += pure[oh + kernel] - pure[oh];
            }
        }

        if (openPad >= 0)
        {
            padStart[padRanges] = openPad;
            padEnd[padRanges] = StandardV33Model.OutH;
            padRanges++;
        }

        if (openActive >= 0)
        {
            activeStart[activeRanges] = openActive;
            activeEnd[activeRanges] = StandardV33Model.OutH;
            activeRanges++;
        }
    }

    /// <summary>
    /// One read of the padding weights for every shortcut row in the batch.
    /// Each sample keeps its own 5×512 response.
    /// </summary>
    private unsafe void BuildPadResponses()
    {
        int filters = StandardV33Model.Filters;
        int kernel = StandardV33Model.Kernel;
        int channels = StandardV33Model.Channels;
        int response = kernel * filters;
        for (int sample = 0; sample < _batch; sample++)
        {
            if (_shortcut[sample] != 0)
            {
                _padT.AsSpan(sample * response, response).Clear();
            }
        }

        Span<float> alpha = stackalloc float[_batch];
        fixed (float* tBase = _padT)
        fixed (float* w0 = _model.PadWeight)
        fixed (float* s0 = _model.PadAct)
        fixed (float* meanBase = _savedMean)
        fixed (float* invBase = _savedInv)
        {
            for (int ic = 0; ic < channels; ic++)
            {
                for (int sample = 0; sample < _batch; sample++)
                {
                    if (_shortcut[sample] == 0)
                    {
                        alpha[sample] = 0f;
                        continue;
                    }

                    float mean = meanBase[(sample * channels) + ic];
                    float inv = invBase[(sample * channels) + ic];
                    alpha[sample] = (s0[ic] - mean) * inv;
                }

                for (int kh = 0; kh < kernel; kh++)
                {
                    float* weight = w0 + (((ic * kernel) + kh) * filters);
                    for (int sample = 0; sample < _batch; sample++)
                    {
                        float scale = alpha[sample];
                        if (scale == 0f)
                        {
                            continue;
                        }

                        Axpy(tBase + (sample * response) + (kh * filters), weight, scale, filters);
                    }
                }
            }
        }
    }

    private unsafe void FillAnalytic(int sample, ReadOnlySpan<int> rangeStart, ReadOnlySpan<int> rangeEnd, int rangeCount)
    {
        int filters = StandardV33Model.Filters;
        int response = StandardV33Model.Kernel * filters;
        Span<float> max = _analytic.AsSpan(sample * filters, filters);
        max.Fill(float.NegativeInfinity);
        fixed (float* t0 = _padT)
        fixed (float* u0 = _model.WeightSum)
        fixed (float* gamma = _model.Ln0Gamma)
        fixed (float* beta = _model.Ln0Beta)
        fixed (float* acc = _padAcc)
        fixed (float* mx = max)
        {
            float* sampleT = t0 + (sample * response);
            for (int r = 0; r < rangeCount; r++)
            {
                for (int oh = rangeStart[r]; oh < rangeEnd[r]; oh++)
                {
                    // out[f] = Σ_kh gamma[oh+kh] * T[kh,f] + beta[oh+kh] * U[kh,f]
                    // T and U already summed the 256 channels, so a padding window is five taps.
                    SetFma(acc, sampleT, gamma[oh], u0, beta[oh], filters);
                    for (int kh = 1; kh < StandardV33Model.Kernel; kh++)
                    {
                        int g = oh + kh;
                        Axpy(acc, sampleT + (kh * filters), gamma[g], filters);
                        Axpy(acc, u0 + (kh * filters), beta[g], filters);
                    }

                    int f = 0;
                    if (Avx512F.IsSupported)
                    {
                        for (; f + 16 <= filters; f += 16)
                        {
                            Vector512<float> value = Avx512F.Max(Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(mx + f)), Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(acc + f)));
                            value.StoreUnsafe(ref Unsafe.AsRef<float>(mx + f));
                        }
                    }
                    else if (Avx2.IsSupported)
                    {
                        for (; f + 8 <= filters; f += 8)
                        {
                            Vector256<float> value = Avx.Max(Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(mx + f)), Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(acc + f)));
                            value.StoreUnsafe(ref Unsafe.AsRef<float>(mx + f));
                        }
                    }

                    for (; f < filters; f++)
                    {
                        mx[f] = MathF.Max(mx[f], acc[f]);
                    }
                }
            }
        }
    }

    private static unsafe void SetFma(float* dst, float* a, float aScale, float* b, float bScale, int length)
    {
        int i = 0;
        if (Avx512F.IsSupported)
        {
            Vector512<float> aFactor = Vector512.Create(aScale);
            Vector512<float> bFactor = Vector512.Create(bScale);
            for (; i + 16 <= length; i += 16)
            {
                Vector512<float> product = Avx512F.Multiply(aFactor, Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(a + i)));
                Vector512<float> value = Avx512F.FusedMultiplyAdd(bFactor, Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(b + i)), product);
                value.StoreUnsafe(ref Unsafe.AsRef<float>(dst + i));
            }
        }
        else if (Avx2.IsSupported && Fma.IsSupported)
        {
            Vector256<float> aFactor = Vector256.Create(aScale);
            Vector256<float> bFactor = Vector256.Create(bScale);
            for (; i + 8 <= length; i += 8)
            {
                Vector256<float> product = Avx.Multiply(aFactor, Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(a + i)));
                Vector256<float> value = Fma.MultiplyAdd(bFactor, Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(b + i)), product);
                value.StoreUnsafe(ref Unsafe.AsRef<float>(dst + i));
            }
        }

        for (; i < length; i++)
        {
            dst[i] = (aScale * a[i]) + (bScale * b[i]);
        }
    }

    private static unsafe void Axpy(float* dst, float* src, float scale, int length)
    {
        if (scale == 0f)
        {
            return;
        }

        int i = 0;
        if (Avx512F.IsSupported)
        {
            Vector512<float> factor = Vector512.Create(scale);
            for (; i + 16 <= length; i += 16)
            {
                Vector512<float> value = Avx512F.FusedMultiplyAdd(
                    factor,
                    Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(src + i)),
                    Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(dst + i)));
                value.StoreUnsafe(ref Unsafe.AsRef<float>(dst + i));
            }
        }
        else if (Avx2.IsSupported && Fma.IsSupported)
        {
            Vector256<float> factor = Vector256.Create(scale);
            for (; i + 8 <= length; i += 8)
            {
                Vector256<float> value = Fma.MultiplyAdd(
                    factor,
                    Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(src + i)),
                    Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(dst + i)));
                value.StoreUnsafe(ref Unsafe.AsRef<float>(dst + i));
            }
        }
        else if (Vector.IsHardwareAccelerated)
        {
            int width = Vector<float>.Count;
            var factor = new Vector<float>(scale);
            for (; i + width <= length; i += width)
            {
                Vector<float> value = Vector.LoadUnsafe(ref Unsafe.AsRef<float>(dst), (nuint)i)
                    + (factor * Vector.LoadUnsafe(ref Unsafe.AsRef<float>(src), (nuint)i));
                value.StoreUnsafe(ref Unsafe.AsRef<float>(dst), (nuint)i);
            }
        }

        for (; i < length; i++)
        {
            dst[i] += scale * src[i];
        }
    }

    private void CombineAnalytic(int sample)
    {
        int filters = StandardV33Model.Filters;
        int offset = sample * filters;
        for (int f = 0; f < filters; f++)
        {
            _pooled[offset + f] = MathF.Max(_pooled[offset + f], _analytic[offset + f]);
        }
    }

    private void ConvolveRanges(int sample, ReadOnlySpan<int> rangeStart, ReadOnlySpan<int> rangeEnd)
    {
        int nr = _model.Nr;
        int n = 0;
        for (int r = 0; r < rangeStart.Length; r++)
        {
            int start = rangeStart[r];
            int end = rangeEnd[r];
            if (end <= start)
            {
                continue;
            }

            int first = start / nr;
            int last = (end - 1) / nr;
            for (int tile = first; tile <= last; tile++)
            {
                int oh0 = tile * nr;
                int lo = Math.Max(0, start - oh0);
                int hi = Math.Min(nr, end - oh0);
                hi = Math.Min(hi, StandardV33Model.OutH - oh0);
                if (hi <= lo)
                {
                    continue;
                }

                _tileOh[n] = oh0;
                _tileLo[n] = lo;
                _tileHi[n] = hi;
                n++;
            }
        }

        _pooled.AsSpan(sample * StandardV33Model.Filters, StandardV33Model.Filters).Fill(float.NegativeInfinity);
        if (n == 0)
        {
            return;
        }

        _shape = ConvShape.Listed;
        _geomSample = sample;
        _geomTiles = n;
        RunGeometry();
    }

    private void RunGeometry()
    {
        int threads = ChooseThreads(StandardV33Model.Filters / StandardV33Model.Mr);
        if (_geomTiles < 4)
        {
            threads = 1;
        }

        _threads = threads;
        if (threads == 1)
        {
            ConvolveWorker(0);
        }
        else
        {
            Parallel.For(0, threads, _convolve);
        }
    }

    private void ConvolveWorker(int id)
    {
        int tiles = StandardV33Model.Filters / StandardV33Model.Mr;
        int start = (int)((long)id * tiles / _threads);
        int end = (int)((long)(id + 1) * tiles / _threads);
        ConvolveFilters(start, end);
    }

    /// <summary>
    /// Each worker owns a slice of filters. A column tile reuses its conv window across those filters,
    /// loaded straight from the channel rows. The tile accumulator is <c>Filters × Nr</c>, not the spatial map.
    /// </summary>
    private unsafe void ConvolveFilters(int tileStart, int tileEnd)
    {
        if (tileStart == tileEnd)
        {
            return;
        }

        int nr = _model.Nr;
        int rowStride = StandardV33Model.RowStride;
        int channels = StandardV33Model.Channels;
        int kernel = StandardV33Model.Kernel;
        int sampleStride = channels * rowStride;
        int tilesPer = _model.TilesPerSample;
        int filters = StandardV33Model.Filters;
        int depth = StandardV33Model.Depth;
        int mr = StandardV33Model.Mr;
        fixed (float* a0 = _model.PackedA)
        fixed (float* convRaw = _conv)
        fixed (float* tileRaw = _tileRaw)
        fixed (float* pooled = _pooled)
        {
            float* conv0 = Align64(convRaw);
            float* cBase = Align64(tileRaw);
            for (int col = 0; col < _geomTiles; col++)
            {
                int sample;
                int oh0;
                int lo;
                int hi;
                if (_shape == ConvShape.Listed)
                {
                    sample = _geomSample;
                    oh0 = _tileOh[col];
                    lo = _tileLo[col];
                    hi = _tileHi[col];
                }
                else if (_shape == ConvShape.Single)
                {
                    sample = _geomSample;
                    oh0 = col * nr;
                    lo = 0;
                    hi = Math.Min(nr, StandardV33Model.OutH - oh0);
                }
                else
                {
                    sample = col / tilesPer;
                    oh0 = (col - (sample * tilesPer)) * nr;
                    lo = 0;
                    hi = Math.Min(nr, StandardV33Model.OutH - oh0);
                }

                float* sampleConv = conv0 + (sample * sampleStride);
                for (int c0 = 0; c0 < channels; c0 += ChannelBlock)
                {
                    int nch = Math.Min(ChannelBlock, channels - c0);
                    bool accumulate = c0 != 0;
                    float* rows = sampleConv + (c0 * rowStride) + oh0;
                    for (int filterTile = tileStart; filterTile < tileEnd; filterTile++)
                    {
                        float* a = a0 + (((filterTile * depth) + (c0 * kernel)) * mr);
                        float* c = cBase + (filterTile * mr * nr);
                        if (nr == 32)
                        {
                            Kernel512_8x32(a, rows, rowStride, nch, c, nr, accumulate);
                        }
                        else
                        {
                            Kernel256_8x16(a, rows, rowStride, nch, c, nr, accumulate);
                        }
                    }
                }

                int filter0 = tileStart * mr;
                int filter1 = tileEnd * mr;
                for (int filter = filter0; filter < filter1; filter++)
                {
                    float lane = MaxPrefix(cBase + (filter * nr) + lo, hi - lo);
                    int slot = (sample * filters) + filter;
                    pooled[slot] = MathF.Max(pooled[slot], lane);
                }
            }
        }
    }

    private unsafe void ConvolvePortable(int sample, ReadOnlySpan<int> rangeStart, ReadOnlySpan<int> rangeEnd)
    {
        bool full = rangeStart.IsEmpty;
        int filters = StandardV33Model.Filters;
        int channels = StandardV33Model.Channels;
        int kernel = StandardV33Model.Kernel;
        int depth = StandardV33Model.Depth;
        int stride = StandardV33Model.RowStride;
        fixed (float* convRaw = _conv)
        fixed (float* weight = _model.ConvWeight)
        fixed (float* acc = _portAcc)
        fixed (float* pooled = _pooled)
        {
            float* conv = Align64(convRaw) + (sample * channels * stride);
            for (int filter = 0; filter < filters; filter++)
            {
                float* w = weight + (filter * depth);
                if (full)
                {
                    new Span<float>(acc, StandardV33Model.OutH).Clear();
                    for (int ic = 0; ic < channels; ic++)
                    {
                        for (int kh = 0; kh < kernel; kh++)
                        {
                            Axpy(acc, conv + (ic * stride) + kh, w[(ic * kernel) + kh], StandardV33Model.OutH);
                        }
                    }

                    pooled[(sample * filters) + filter] = MaxPrefix(acc, StandardV33Model.OutH);
                }
                else
                {
                    float max = float.NegativeInfinity;
                    for (int r = 0; r < rangeStart.Length; r++)
                    {
                        int start = rangeStart[r];
                        int length = rangeEnd[r] - start;
                        if (length <= 0)
                        {
                            continue;
                        }

                        new Span<float>(acc, length).Clear();
                        for (int ic = 0; ic < channels; ic++)
                        {
                            for (int kh = 0; kh < kernel; kh++)
                            {
                                Axpy(acc, conv + (ic * stride) + start + kh, w[(ic * kernel) + kh], length);
                            }
                        }

                        max = MathF.Max(max, MaxPrefix(acc, length));
                    }

                    pooled[(sample * filters) + filter] = max;
                }
            }
        }
    }

    private void ActivatePooled()
    {
        ReadOnlySpan<float> bias = _model.ConvBias;
        for (int sample = 0; sample < _batch; sample++)
        {
            Span<float> row = _pooled.AsSpan(sample * StandardV33Model.Filters, StandardV33Model.Filters);
            for (int i = 0; i < row.Length; i++)
            {
                row[i] += bias[i];
            }

            SimdMath.Gelu(row, _model.GeluCubic, _model.GeluScale, _model.GeluOne, _model.GeluHalf);
        }
    }

    private void Project(int sample, Span<float> dest)
    {
        LayerNorm1(_pooled.AsSpan(sample * StandardV33Model.Filters, StandardV33Model.Filters), _norm);
        DenseAndSoftmax(_norm, dest);
    }

    private void LayerNorm1(ReadOnlySpan<float> source, Span<float> dest)
    {
        float sum = 0f;
        float sumSq = 0f;
        for (int i = 0; i < StandardV33Model.Filters; i++)
        {
            float value = source[i];
            sum += value;
            sumSq += value * value;
        }

        float mean = sum * _model.InvLength;
        float variance = MathF.Max((sumSq * _model.InvLength) - (mean * mean), 0f);
        float invStd = 1f / MathF.Sqrt(variance + _model.Epsilon);
        ReadOnlySpan<float> gamma = _model.Ln1Gamma;
        ReadOnlySpan<float> beta = _model.Ln1Beta;
        for (int i = 0; i < StandardV33Model.Filters; i++)
        {
            float scale = invStd * gamma[i];
            dest[i] = ((source[i] - mean) * scale) + beta[i];
        }
    }

    private unsafe void DenseAndSoftmax(ReadOnlySpan<float> features, Span<float> dest)
    {
        _model.DenseBias.AsSpan(0, StandardV33Model.Classes).CopyTo(_logits);
        fixed (float* y = features)
        fixed (float* w = _model.DenseWeight)
        fixed (float* acc = _logits)
        {
            for (int p = 0; p < StandardV33Model.Filters; p++)
            {
                float weight = y[p];
                float* row = w + (p * StandardV33Model.Classes);
                int j = 0;
                if (Avx512F.IsSupported)
                {
                    Vector512<float> broadcast = Vector512.Create(weight);
                    for (; j + 16 <= StandardV33Model.Classes; j += 16)
                    {
                        Vector512<float> value = Avx512F.FusedMultiplyAdd(
                            broadcast,
                            Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(row + j)),
                            Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(acc + j)));
                        value.StoreUnsafe(ref Unsafe.AsRef<float>(acc + j));
                    }
                }
                else if (Avx2.IsSupported && Fma.IsSupported)
                {
                    Vector256<float> broadcast = Vector256.Create(weight);
                    for (; j + 8 <= StandardV33Model.Classes; j += 8)
                    {
                        Vector256<float> value = Fma.MultiplyAdd(
                            broadcast,
                            Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(row + j)),
                            Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(acc + j)));
                        value.StoreUnsafe(ref Unsafe.AsRef<float>(acc + j));
                    }
                }

                for (; j < StandardV33Model.Classes; j++)
                {
                    acc[j] += weight * row[j];
                }
            }
        }

        Span<float> logits = _logits.AsSpan(0, StandardV33Model.Classes);
        float max = float.NegativeInfinity;
        for (int j = 0; j < logits.Length; j++)
        {
            max = MathF.Max(max, logits[j]);
        }

        if (float.IsNegativeInfinity(max))
        {
            max = 0f;
        }

        float sum = 0f;
        for (int j = 0; j < logits.Length; j++)
        {
            float exp = MathF.Exp(logits[j] - max);
            logits[j] = exp;
            sum += exp;
        }

        float inv = 1f / sum;
        for (int j = 0; j < logits.Length; j++)
        {
            dest[j] = logits[j] * inv;
        }
    }

    private static int ChooseThreads(int units)
    {
        int threads = Environment.ProcessorCount;
        if (SimdGemm.ThreadBudget > 0)
        {
            threads = Math.Min(threads, SimdGemm.ThreadBudget);
        }

        if (units < 2 || threads < 2)
        {
            return 1;
        }

        return Math.Min(threads, units);
    }

    private static unsafe float* Align64(float* raw)
    {
        nint address = (nint)raw;
        nint aligned = (address + 63) & ~63;
        return (float*)aligned;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel512_8x32(float* a, float* row0, int rowStride, int channels, float* c, int ldc, bool accumulate)
    {
        Vector512<float> r00;
        Vector512<float> r01;
        Vector512<float> r10;
        Vector512<float> r11;
        Vector512<float> r20;
        Vector512<float> r21;
        Vector512<float> r30;
        Vector512<float> r31;
        Vector512<float> r40;
        Vector512<float> r41;
        Vector512<float> r50;
        Vector512<float> r51;
        Vector512<float> r60;
        Vector512<float> r61;
        Vector512<float> r70;
        Vector512<float> r71;
        if (accumulate)
        {
            r00 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c));
            r01 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c), 16);
            r10 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + ldc));
            r11 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + ldc), 16);
            r20 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
            r21 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)), 16);
            r30 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
            r31 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)), 16);
            r40 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
            r41 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)), 16);
            r50 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
            r51 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)), 16);
            r60 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
            r61 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)), 16);
            r70 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
            r71 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)), 16);
        }
        else
        {
            r00 = Vector512<float>.Zero;
            r01 = Vector512<float>.Zero;
            r10 = Vector512<float>.Zero;
            r11 = Vector512<float>.Zero;
            r20 = Vector512<float>.Zero;
            r21 = Vector512<float>.Zero;
            r30 = Vector512<float>.Zero;
            r31 = Vector512<float>.Zero;
            r40 = Vector512<float>.Zero;
            r41 = Vector512<float>.Zero;
            r50 = Vector512<float>.Zero;
            r51 = Vector512<float>.Zero;
            r60 = Vector512<float>.Zero;
            r61 = Vector512<float>.Zero;
            r70 = Vector512<float>.Zero;
            r71 = Vector512<float>.Zero;
        }

        for (int ic = 0; ic < channels; ic++)
        {
            float* row = row0 + (ic * rowStride);
            for (int kh = 0; kh < StandardV33Model.Kernel; kh++)
            {
                Vector512<float> b0 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(row + kh));
                Vector512<float> b1 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(row + kh), 16);
                r00 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[0]), b0, r00);
                r01 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[0]), b1, r01);
                r10 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[1]), b0, r10);
                r11 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[1]), b1, r11);
                r20 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[2]), b0, r20);
                r21 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[2]), b1, r21);
                r30 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[3]), b0, r30);
                r31 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[3]), b1, r31);
                r40 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[4]), b0, r40);
                r41 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[4]), b1, r41);
                r50 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[5]), b0, r50);
                r51 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[5]), b1, r51);
                r60 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[6]), b0, r60);
                r61 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[6]), b1, r61);
                r70 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[7]), b0, r70);
                r71 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[7]), b1, r71);
                a += 8;
            }
        }

        r00.StoreUnsafe(ref Unsafe.AsRef<float>(c));
        r01.StoreUnsafe(ref Unsafe.AsRef<float>(c), 16);
        r10.StoreUnsafe(ref Unsafe.AsRef<float>(c + ldc));
        r11.StoreUnsafe(ref Unsafe.AsRef<float>(c + ldc), 16);
        r20.StoreUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
        r21.StoreUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)), 16);
        r30.StoreUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
        r31.StoreUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)), 16);
        r40.StoreUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
        r41.StoreUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)), 16);
        r50.StoreUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
        r51.StoreUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)), 16);
        r60.StoreUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
        r61.StoreUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)), 16);
        r70.StoreUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
        r71.StoreUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)), 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel256_8x16(float* a, float* row0, int rowStride, int channels, float* c, int ldc, bool accumulate)
    {
        Vector256<float> r00;
        Vector256<float> r01;
        Vector256<float> r10;
        Vector256<float> r11;
        Vector256<float> r20;
        Vector256<float> r21;
        Vector256<float> r30;
        Vector256<float> r31;
        Vector256<float> r40;
        Vector256<float> r41;
        Vector256<float> r50;
        Vector256<float> r51;
        Vector256<float> r60;
        Vector256<float> r61;
        Vector256<float> r70;
        Vector256<float> r71;
        if (accumulate)
        {
            r00 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c));
            r01 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c), 8);
            r10 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + ldc));
            r11 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + ldc), 8);
            r20 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
            r21 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)), 8);
            r30 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
            r31 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)), 8);
            r40 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
            r41 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)), 8);
            r50 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
            r51 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)), 8);
            r60 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
            r61 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)), 8);
            r70 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
            r71 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)), 8);
        }
        else
        {
            r00 = Vector256<float>.Zero;
            r01 = Vector256<float>.Zero;
            r10 = Vector256<float>.Zero;
            r11 = Vector256<float>.Zero;
            r20 = Vector256<float>.Zero;
            r21 = Vector256<float>.Zero;
            r30 = Vector256<float>.Zero;
            r31 = Vector256<float>.Zero;
            r40 = Vector256<float>.Zero;
            r41 = Vector256<float>.Zero;
            r50 = Vector256<float>.Zero;
            r51 = Vector256<float>.Zero;
            r60 = Vector256<float>.Zero;
            r61 = Vector256<float>.Zero;
            r70 = Vector256<float>.Zero;
            r71 = Vector256<float>.Zero;
        }

        for (int ic = 0; ic < channels; ic++)
        {
            float* row = row0 + (ic * rowStride);
            for (int kh = 0; kh < StandardV33Model.Kernel; kh++)
            {
                Vector256<float> b0 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(row + kh));
                Vector256<float> b1 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(row + kh), 8);
                r00 = Fma.MultiplyAdd(Vector256.Create(a[0]), b0, r00);
                r01 = Fma.MultiplyAdd(Vector256.Create(a[0]), b1, r01);
                r10 = Fma.MultiplyAdd(Vector256.Create(a[1]), b0, r10);
                r11 = Fma.MultiplyAdd(Vector256.Create(a[1]), b1, r11);
                r20 = Fma.MultiplyAdd(Vector256.Create(a[2]), b0, r20);
                r21 = Fma.MultiplyAdd(Vector256.Create(a[2]), b1, r21);
                r30 = Fma.MultiplyAdd(Vector256.Create(a[3]), b0, r30);
                r31 = Fma.MultiplyAdd(Vector256.Create(a[3]), b1, r31);
                r40 = Fma.MultiplyAdd(Vector256.Create(a[4]), b0, r40);
                r41 = Fma.MultiplyAdd(Vector256.Create(a[4]), b1, r41);
                r50 = Fma.MultiplyAdd(Vector256.Create(a[5]), b0, r50);
                r51 = Fma.MultiplyAdd(Vector256.Create(a[5]), b1, r51);
                r60 = Fma.MultiplyAdd(Vector256.Create(a[6]), b0, r60);
                r61 = Fma.MultiplyAdd(Vector256.Create(a[6]), b1, r61);
                r70 = Fma.MultiplyAdd(Vector256.Create(a[7]), b0, r70);
                r71 = Fma.MultiplyAdd(Vector256.Create(a[7]), b1, r71);
                a += 8;
            }
        }

        r00.StoreUnsafe(ref Unsafe.AsRef<float>(c));
        r01.StoreUnsafe(ref Unsafe.AsRef<float>(c), 8);
        r10.StoreUnsafe(ref Unsafe.AsRef<float>(c + ldc));
        r11.StoreUnsafe(ref Unsafe.AsRef<float>(c + ldc), 8);
        r20.StoreUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
        r21.StoreUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)), 8);
        r30.StoreUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
        r31.StoreUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)), 8);
        r40.StoreUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
        r41.StoreUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)), 8);
        r50.StoreUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
        r51.StoreUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)), 8);
        r60.StoreUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
        r61.StoreUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)), 8);
        r70.StoreUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
        r71.StoreUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)), 8);
    }

    private static unsafe float MaxPrefix(float* row, int length)
    {
        float max = float.NegativeInfinity;
        int j = 0;
        if (Avx512F.IsSupported)
        {
            Vector512<float> acc = Vector512.Create(float.NegativeInfinity);
            for (; j + 16 <= length; j += 16)
            {
                acc = Avx512F.Max(acc, Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(row + j)));
            }

            max = HorizontalMax(acc);
        }
        else if (Avx2.IsSupported)
        {
            Vector256<float> acc = Vector256.Create(float.NegativeInfinity);
            for (; j + 8 <= length; j += 8)
            {
                acc = Avx.Max(acc, Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(row + j)));
            }

            max = HorizontalMax(acc);
        }

        for (; j < length; j++)
        {
            max = MathF.Max(max, row[j]);
        }

        return max;
    }

    private static float HorizontalMax(Vector512<float> value)
    {
        Vector256<float> lower = value.GetLower();
        Vector256<float> upper = value.GetUpper();
        return HorizontalMax(Avx.Max(lower, upper));
    }

    private static float HorizontalMax(Vector256<float> value)
    {
        Vector128<float> lower = value.GetLower();
        Vector128<float> upper = value.GetUpper();
        Vector128<float> max = Sse.Max(lower, upper);
        max = Sse.Max(max, Sse.Shuffle(max, max, 0b_01_00_11_10));
        max = Sse.Max(max, Sse.Shuffle(max, max, 0b_10_11_00_01));
        return max.ToScalar();
    }

    private enum ConvShape
    {
        Dense,
        Single,
        Listed,
    }

    private interface ITokenReader
    {
        int At(int index);
    }

    private readonly unsafe struct IntTokens(int* pointer) : ITokenReader
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int At(int index) => pointer[index];
    }

    private readonly unsafe struct UShortTokens(ushort* pointer) : ITokenReader
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int At(int index) => pointer[index];
    }
}
