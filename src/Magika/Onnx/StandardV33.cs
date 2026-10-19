using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Magika.Onnx;

/// <summary>
/// Weights for the bundled <c>standard_v3_3</c> graph.
/// The network is fixed: a 64-d byte embedding, GELU, a length-512 layer norm,
/// one 5×1 convolution (256 channels to 512 filters), GELU, global max, a second
/// layer norm, and a 512×214 dense layer with softmax.
/// </summary>
internal sealed class StandardV33Model
{
    internal const int Sequence = 2048;
    internal const int EmbedDim = 64;
    internal const int Vocab = 257;
    internal const int Groups = 512;
    internal const int Channels = 256;
    internal const int Kernel = 5;
    internal const int Filters = 512;
    internal const int Depth = Channels * Kernel;
    internal const int OutH = Groups - Kernel + 1;
    internal const int Classes = 214;
    internal const int Mr = 8;
    internal const int SampleStride = 512;
    internal const int PadToken = 256;
    internal const int TokensPerGroup = Channels / EmbedDim;
    internal const int MinPureWindows = 64;

    private const string EmbedName = "jax2tf_get_logits_/Const:0";
    private const string EmbedBiasName = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/Dense_0/Reshape:0";
    private const string Ln0GammaName = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_0/Reshape_2:0";
    private const string Ln0BetaName = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_0/Reshape_3:0";
    private const string ConvWeightName = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/Conv_0/transpose_3:0";
    private const string ConvBiasName = "const_fold_opt__209";
    private const string Ln1GammaName = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_1/Reshape_2:0";
    private const string Ln1BetaName = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_1/Reshape_3:0";
    private const string DenseWeightName = "jax2tf_get_logits_/Const_24:0";
    private const string DenseBiasName = "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/Dense_1/Reshape:0";
    private const string GeluCubicName = "jax2tf_get_logits_/Const_2:0";
    private const string GeluScaleName = "jax2tf_get_logits_/Const_15:0";
    private const string GeluHalfName = "jax2tf_get_logits_/Const_17:0";
    private const string GeluOneName = "jax2tf_get_logits_/Const_4:0";
    private const string EpsilonName = "jax2tf_get_logits_/Const_21:0";
    private const string InvLengthName = "ConstantFolding/jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_1/truediv_recip:0";

    private StandardV33Model(
        float[] embed,
        float[] embedBias,
        float[] ln0Gamma,
        float[] ln0Beta,
        float[] convWeight,
        float[] convBias,
        float[] ln1Gamma,
        float[] ln1Beta,
        float[] denseWeight,
        float[] denseBias,
        float[] packedA,
        int nr,
        float[] padAct,
        float[] padWeight,
        float[] weightSum,
        float geluCubic,
        float geluScale,
        float geluHalf,
        float geluOne,
        float epsilon,
        float invLength)
    {
        Embed = embed;
        EmbedBias = embedBias;
        Ln0Gamma = ln0Gamma;
        Ln0Beta = ln0Beta;
        ConvWeight = convWeight;
        ConvBias = convBias;
        Ln1Gamma = ln1Gamma;
        Ln1Beta = ln1Beta;
        DenseWeight = denseWeight;
        DenseBias = denseBias;
        PackedA = packedA;
        Nr = nr;
        PadAct = padAct;
        PadWeight = padWeight;
        WeightSum = weightSum;
        GeluCubic = geluCubic;
        GeluScale = geluScale;
        GeluHalf = geluHalf;
        GeluOne = geluOne;
        Epsilon = epsilon;
        InvLength = invLength;
    }

    internal float[] Embed { get; }
    internal float[] EmbedBias { get; }
    internal float[] Ln0Gamma { get; }
    internal float[] Ln0Beta { get; }
    internal float[] ConvWeight { get; }
    internal float[] ConvBias { get; }
    internal float[] Ln1Gamma { get; }
    internal float[] Ln1Beta { get; }
    internal float[] DenseWeight { get; }
    internal float[] DenseBias { get; }

    /// <summary>Conv filters packed for the 8-row microkernel: each K step holds 8 contiguous weights.</summary>
    internal float[] PackedA { get; }

    /// <summary>GELU(embed(pad) + bias), repeated across the four tokens of a pure padding group.</summary>
    internal float[] PadAct { get; }

    /// <summary>Conv weights as [channel, tap, filter] so a padding response is a short axpy.</summary>
    internal float[] PadWeight { get; }

    /// <summary>Sum of conv weights over channels, laid out [tap, filter].</summary>
    internal float[] WeightSum { get; }

    internal int Nr { get; }
    internal int TilesPerSample => SampleStride / Nr;
    internal bool Portable => PackedA.Length == 0;
    internal float GeluCubic { get; }
    internal float GeluScale { get; }
    internal float GeluHalf { get; }
    internal float GeluOne { get; }
    internal float Epsilon { get; }
    internal float InvLength { get; }

    internal static StandardV33Model? TryCreate(OnnxModel model)
    {
        if (!HasConvSignature(model))
        {
            return null;
        }

        Dictionary<string, OnnxTensor> init = model.Initializers;
        float[]? embed = Matrix(init, EmbedName, Vocab, EmbedDim);
        float[]? embedBias = Matrix(init, EmbedBiasName, 1, 1, EmbedDim);
        float[]? ln0Gamma = Matrix(init, Ln0GammaName, 1, Groups, 1);
        float[]? ln0Beta = Matrix(init, Ln0BetaName, 1, Groups, 1);
        float[]? convWeight = Matrix(init, ConvWeightName, Filters, Channels, Kernel, 1);
        float[]? convBias = Matrix(init, ConvBiasName, 1, Filters, 1);
        float[]? ln1Gamma = Matrix(init, Ln1GammaName, 1, Filters);
        float[]? ln1Beta = Matrix(init, Ln1BetaName, 1, Filters);
        float[]? denseWeight = Matrix(init, DenseWeightName, Filters, Classes);
        float[]? denseBias = Matrix(init, DenseBiasName, 1, Classes);
        if (embed is null || embedBias is null || ln0Gamma is null || ln0Beta is null
            || convWeight is null || convBias is null || ln1Gamma is null || ln1Beta is null
            || denseWeight is null || denseBias is null)
        {
            return null;
        }

        if (!Scalar(init, GeluCubicName, out float geluCubic)
            || !Scalar(init, GeluScaleName, out float geluScale)
            || !Scalar(init, GeluHalfName, out float geluHalf)
            || !Scalar(init, GeluOneName, out float geluOne)
            || !Scalar(init, EpsilonName, out float epsilon)
            || !Scalar(init, InvLengthName, out float invLength))
        {
            return null;
        }

        if (geluHalf != 0.5f || geluOne != 1f || invLength != 1f / Groups || epsilon <= 0f || geluScale == 0f)
        {
            return null;
        }

        int nr = Avx512F.IsSupported ? 32 : Avx2.IsSupported && Fma.IsSupported ? 16 : 0;
        float[] packedA = nr == 0 ? [] : PackA(convWeight);
        PackPadTables(convWeight, embed, embedBias, geluCubic, geluScale, geluOne, geluHalf, out float[] padAct, out float[] padWeight, out float[] weightSum);
        return new StandardV33Model(
            embed,
            embedBias,
            ln0Gamma,
            ln0Beta,
            convWeight,
            convBias,
            ln1Gamma,
            ln1Beta,
            denseWeight,
            denseBias,
            packedA,
            nr == 0 ? 1 : nr,
            padAct,
            padWeight,
            weightSum,
            geluCubic,
            geluScale,
            geluHalf,
            geluOne,
            epsilon,
            invLength);
    }

    private static bool HasConvSignature(OnnxModel model)
    {
        bool conv = false;
        bool pool = false;
        foreach (OnnxNode node in model.Nodes)
        {
            if (node.Op == "GlobalMaxPool")
            {
                pool = true;
            }

            if (node.Op != "Conv")
            {
                continue;
            }

            if ((int)node.Int("group", 1) != 1)
            {
                return false;
            }

            int[] strides = node.Ints("strides");
            if (strides.Length > 0 && (strides[0] != 1 || (strides.Length > 1 && strides[1] != 1)))
            {
                return false;
            }

            int[] dilations = node.Ints("dilations");
            if (dilations.Length > 0 && (dilations[0] != 1 || (dilations.Length > 1 && dilations[1] != 1)))
            {
                return false;
            }

            foreach (int pad in node.Ints("pads"))
            {
                if (pad != 0)
                {
                    return false;
                }
            }

            int[] kernel = node.Ints("kernel_shape");
            if (kernel.Length >= 2 && (kernel[0] != Kernel || kernel[1] != 1))
            {
                return false;
            }

            conv = true;
        }

        return conv && pool;
    }

    private static float[]? Matrix(Dictionary<string, OnnxTensor> init, string name, params int[] shape)
    {
        if (!init.TryGetValue(name, out OnnxTensor? tensor) || tensor.Floats is null || tensor.Shape.Length != shape.Length)
        {
            return null;
        }

        for (int i = 0; i < shape.Length; i++)
        {
            if (tensor.Shape[i] != shape[i])
            {
                return null;
            }
        }

        return tensor.Floats;
    }

    private static bool Scalar(Dictionary<string, OnnxTensor> init, string name, out float value)
    {
        value = 0f;
        if (!init.TryGetValue(name, out OnnxTensor? tensor) || tensor.Floats is null || tensor.Length != 1)
        {
            return false;
        }

        value = tensor.Floats[0];
        return true;
    }

    private static float[] PackA(float[] weight)
    {
        float[] packed = GC.AllocateUninitializedArray<float>(checked((Filters / Mr) * Depth * Mr));
        int tileCount = Filters / Mr;
        for (int tile = 0; tile < tileCount; tile++)
        {
            for (int p = 0; p < Depth; p++)
            {
                int dst = ((tile * Depth) + p) * Mr;
                int row = tile * Mr;
                for (int r = 0; r < Mr; r++)
                {
                    packed[dst + r] = weight[((row + r) * Depth) + p];
                }
            }
        }

        return packed;
    }

    private static void PackPadTables(
        float[] weight,
        float[] embed,
        float[] embedBias,
        float cubic,
        float scale,
        float one,
        float half,
        out float[] padAct,
        out float[] padWeight,
        out float[] weightSum)
    {
        float[] row = new float[EmbedDim];
        int pad = PadToken * EmbedDim;
        for (int d = 0; d < EmbedDim; d++)
        {
            row[d] = embed[pad + d] + embedBias[d];
        }

        SimdMath.Gelu(row, cubic, scale, one, half);
        padAct = new float[Channels];
        for (int c = 0; c < Channels; c++)
        {
            padAct[c] = row[c % EmbedDim];
        }

        padWeight = new float[Channels * Kernel * Filters];
        weightSum = new float[Kernel * Filters];
        for (int f = 0; f < Filters; f++)
        {
            for (int ic = 0; ic < Channels; ic++)
            {
                for (int kh = 0; kh < Kernel; kh++)
                {
                    float value = weight[(((f * Channels) + ic) * Kernel) + kh];
                    padWeight[(((ic * Kernel) + kh) * Filters) + f] = value;
                    weightSum[(kh * Filters) + f] += value;
                }
            }
        }
    }
}

/// <summary>
/// Per-lane activation state for <see cref="StandardV33Model"/>. Not thread-safe.
/// One evaluation at a time, matching <see cref="OnnxSession"/>.
/// </summary>
internal sealed class StandardV33Session
{
    private readonly StandardV33Model _model;
    private readonly Action<int> _convolve;
    private int _batch;
    private int _threads;
    private int _capacity;
    private int[] _tokens = [];
    private float[] _act = [];
    private float[] _conv = [];
    private float[] _packedB = [];
    private float[] _convOut = [];
    private float[] _pooled = [];
    private float[] _norm = new float[StandardV33Model.Filters];
    private float[] _logits = new float[StandardV33Model.Classes];
    private float[] _tiles = [];
    private readonly float[] _mean = new float[StandardV33Model.Channels];
    private readonly float[] _invStd = new float[StandardV33Model.Channels];
    private readonly float[] _padT = new float[StandardV33Model.Kernel * StandardV33Model.Filters];
    private readonly float[] _padAcc = new float[StandardV33Model.Filters];
    private readonly int[] _activeOh = new int[StandardV33Model.OutH];
    private byte[] _shortcut = [];
    private float[] _savedMean = [];
    private float[] _savedInv = [];
    private float[] _analytic = [];
    private int _geomTiles;
    private int _geomPad;
    private int _geomColumns;
    private int _geomSample;
    private bool _geomBatch;

    internal static void CheckKernels()
    {
        const int k = 48;
        const int mr = StandardV33Model.Mr;
        float[] a = new float[mr * k];
        var random = new Random(3);
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)((random.NextDouble() * 2) - 1);
        }

        if (Avx512F.IsSupported)
        {
            CheckOne(a, k, nr: 32, avx512: true);
        }

        if (Avx2.IsSupported && Fma.IsSupported)
        {
            CheckOne(a, k, nr: 16, avx512: false);
        }
    }

    private static unsafe void CheckOne(float[] a, int k, int nr, bool avx512)
    {
        float[] b = new float[k * nr];
        float[] c = new float[StandardV33Model.Mr * nr];
        var random = new Random(avx512 ? 5 : 9);
        for (int i = 0; i < b.Length; i++)
        {
            b[i] = (float)((random.NextDouble() * 2) - 1);
        }

        // A is packed k-major, 8-wide, which is how the microkernel reads it.
        // Build that from 8 logical rows stored the same way.
        fixed (float* ap = a)
        fixed (float* bp = b)
        fixed (float* cp = c)
        {
            if (avx512)
            {
                Kernel512_8x32(ap, bp, cp, k, nr, accumulate: false);
            }
            else
            {
                Kernel256_8x16(ap, bp, cp, k, nr, accumulate: false);
            }
        }

        for (int row = 0; row < StandardV33Model.Mr; row++)
        {
            for (int col = 0; col < nr; col++)
            {
                float sum = 0f;
                for (int p = 0; p < k; p++)
                {
                    sum += a[(p * StandardV33Model.Mr) + row] * b[(p * nr) + col];
                }

                float actual = c[(row * nr) + col];
                float delta = MathF.Abs(sum - actual);
                if (delta > 1e-3f)
                {
                    throw new InvalidOperationException($"Kernel mismatch row {row} col {col} delta {delta}.");
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

    internal float[] Evaluate(ReadOnlySpan<int> tokens, int batch, int paddingToken = -1)
    {
        if (batch <= 0 || tokens.Length != batch * StandardV33Model.Sequence)
        {
            throw new ArgumentException("Token buffer does not match the standard_v3_3 sequence.", nameof(tokens));
        }

        Ensure(batch);
        tokens.CopyTo(_tokens);
        _batch = batch;
        bool shortcut = false;
        for (int sample = 0; sample < batch; sample++)
        {
            EmbedAndNormalize(sample);
            bool sparse = IsPadShortcut(sample, paddingToken);
            _shortcut[sample] = sparse ? (byte)1 : (byte)0;
            if (sparse)
            {
                shortcut = true;
                _mean.AsSpan().CopyTo(_savedMean.AsSpan(sample * StandardV33Model.Channels, StandardV33Model.Channels));
                _invStd.AsSpan().CopyTo(_savedInv.AsSpan(sample * StandardV33Model.Channels, StandardV33Model.Channels));
            }
        }

        if (!shortcut)
        {
            if (_model.Portable)
            {
                ConvolvePortable();
            }
            else
            {
                PackB();
                BeginFullGeometry();
                RunGeometry();
            }
        }
        else
        {
            ConvolveShortcutBatch();
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

        int act = batch * StandardV33Model.Groups * StandardV33Model.Channels;
        _tokens = new int[batch * StandardV33Model.Sequence];
        _act = GC.AllocateUninitializedArray<float>(act);
        _conv = GC.AllocateUninitializedArray<float>(act);
        _pooled = GC.AllocateUninitializedArray<float>(batch * StandardV33Model.Filters);
        _shortcut = new byte[batch];
        _savedMean = new float[batch * StandardV33Model.Channels];
        _savedInv = new float[batch * StandardV33Model.Channels];
        _analytic = new float[batch * StandardV33Model.Filters];
        if (!_model.Portable)
        {
            int tiles = batch * _model.TilesPerSample;
            _packedB = GC.AllocateUninitializedArray<float>(tiles * StandardV33Model.Depth * _model.Nr);
            _convOut = GC.AllocateUninitializedArray<float>(StandardV33Model.Filters * batch * StandardV33Model.SampleStride);
            _tiles = [];
        }
        else
        {
            _packedB = GC.AllocateUninitializedArray<float>(batch * StandardV33Model.Depth * StandardV33Model.OutH);
            _tiles = GC.AllocateUninitializedArray<float>(batch * StandardV33Model.Filters * StandardV33Model.OutH);
        }

        _capacity = batch;
    }

    private unsafe void EmbedAndNormalize(int sample)
    {
        int sequence = StandardV33Model.Sequence;
        int dim = StandardV33Model.EmbedDim;
        fixed (float* actBase = _act)
        fixed (float* table = _model.Embed)
        fixed (float* bias = _model.EmbedBias)
        fixed (int* tokenBase = _tokens)
        {
            float* act = actBase + (sample * StandardV33Model.Groups * StandardV33Model.Channels);
            int* tokens = tokenBase + (sample * sequence);
            EmbedRows(tokens, act, table, bias, sequence, dim);
        }

        SimdMath.Gelu(
            _act.AsSpan(sample * sequence * dim, sequence * dim),
            _model.GeluCubic,
            _model.GeluScale,
            _model.GeluOne,
            _model.GeluHalf);
        LayerNorm0(sample);
    }

    private static unsafe void EmbedRows(int* tokens, float* act, float* table, float* bias, int sequence, int dim)
    {
        for (int i = 0; i < sequence; i++)
        {
            int id = tokens[i];
            if ((uint)id >= StandardV33Model.Vocab)
            {
                throw new InvalidDataException($"Embedding index {id} is outside 0..{StandardV33Model.Vocab - 1}.");
            }

            float* src = table + (id * dim);
            float* dst = act + (i * dim);
            if (Avx512F.IsSupported)
            {
                for (int d = 0; d < dim; d += 16)
                {
                    Vector512<float> value = Avx512F.Add(Avx512F.LoadVector512(src + d), Avx512F.LoadVector512(bias + d));
                    Avx512F.Store(dst + d, value);
                }
            }
            else if (Avx2.IsSupported)
            {
                for (int d = 0; d < dim; d += 8)
                {
                    Vector256<float> value = Avx.Add(Avx.LoadVector256(src + d), Avx.LoadVector256(bias + d));
                    Avx.Store(dst + d, value);
                }
            }
            else
            {
                for (int d = 0; d < dim; d++)
                {
                    dst[d] = src[d] + bias[d];
                }
            }
        }
    }

    private unsafe void LayerNorm0(int sample)
    {
        int groups = StandardV33Model.Groups;
        int channels = StandardV33Model.Channels;
        fixed (float* actBase = _act)
        fixed (float* convBase = _conv)
        fixed (float* gamma = _model.Ln0Gamma)
        fixed (float* beta = _model.Ln0Beta)
        fixed (float* mean = _mean)
        fixed (float* invStd = _invStd)
        {
            LayerNorm0Core(
                actBase + (sample * groups * channels),
                convBase + (sample * groups * channels),
                gamma,
                beta,
                mean,
                invStd,
                _model.InvLength,
                _model.Epsilon);
        }
    }

    private static unsafe void LayerNorm0Core(float* act, float* conv, float* gamma, float* beta, float* meanOut, float* invStdOut, float invN, float eps)
    {
        int groups = StandardV33Model.Groups;
        int channels = StandardV33Model.Channels;
        if (Avx512F.IsSupported)
        {
            for (int c = 0; c < channels; c += 16)
            {
                Vector512<float> sum = Vector512<float>.Zero;
                Vector512<float> sumSq = Vector512<float>.Zero;
                for (int g = 0; g < groups; g++)
                {
                    Vector512<float> value = Avx512F.LoadVector512(act + (g * channels) + c);
                    sum = Avx512F.Add(sum, value);
                    sumSq = Avx512F.Add(sumSq, Avx512F.Multiply(value, value));
                }

                Vector512<float> mean = Avx512F.Multiply(sum, Vector512.Create(invN));
                Vector512<float> variance = Avx512F.Subtract(Avx512F.Multiply(sumSq, Vector512.Create(invN)), Avx512F.Multiply(mean, mean));
                variance = Avx512F.Max(variance, Vector512<float>.Zero);
                Vector512<float> invStd = Avx512F.Divide(Vector512<float>.One, Avx512F.Sqrt(Avx512F.Add(variance, Vector512.Create(eps))));
                Avx512F.Store(meanOut + c, mean);
                Avx512F.Store(invStdOut + c, invStd);
                for (int g = 0; g < groups; g++)
                {
                    Vector512<float> value = Avx512F.LoadVector512(act + (g * channels) + c);
                    Vector512<float> scale = Avx512F.Multiply(invStd, Vector512.Create(gamma[g]));
                    Vector512<float> centered = Avx512F.Subtract(value, mean);
                    Vector512<float> y = Avx512F.Add(Avx512F.Multiply(centered, scale), Vector512.Create(beta[g]));
                    Avx512F.Store(act + (g * channels) + c, y);
                }
            }
        }
        else if (Avx2.IsSupported)
        {
            for (int c = 0; c < channels; c += 8)
            {
                Vector256<float> sum = Vector256<float>.Zero;
                Vector256<float> sumSq = Vector256<float>.Zero;
                for (int g = 0; g < groups; g++)
                {
                    Vector256<float> value = Avx.LoadVector256(act + (g * channels) + c);
                    sum = Avx.Add(sum, value);
                    sumSq = Avx.Add(sumSq, Avx.Multiply(value, value));
                }

                Vector256<float> mean = Avx.Multiply(sum, Vector256.Create(invN));
                Vector256<float> variance = Avx.Subtract(Avx.Multiply(sumSq, Vector256.Create(invN)), Avx.Multiply(mean, mean));
                variance = Avx.Max(variance, Vector256<float>.Zero);
                Vector256<float> invStd = Avx.Divide(Vector256<float>.One, Avx.Sqrt(Avx.Add(variance, Vector256.Create(eps))));
                Avx.Store(meanOut + c, mean);
                Avx.Store(invStdOut + c, invStd);
                for (int g = 0; g < groups; g++)
                {
                    Vector256<float> value = Avx.LoadVector256(act + (g * channels) + c);
                    Vector256<float> scale = Avx.Multiply(invStd, Vector256.Create(gamma[g]));
                    Vector256<float> centered = Avx.Subtract(value, mean);
                    Vector256<float> y = Avx.Add(Avx.Multiply(centered, scale), Vector256.Create(beta[g]));
                    Avx.Store(act + (g * channels) + c, y);
                }
            }
        }
        else
        {
            for (int c = 0; c < channels; c++)
            {
                float sum = 0f;
                float sumSq = 0f;
                for (int g = 0; g < groups; g++)
                {
                    float value = act[(g * channels) + c];
                    sum += value;
                    sumSq += value * value;
                }

                float mean = sum * invN;
                float variance = MathF.Max((sumSq * invN) - (mean * mean), 0f);
                float invStd = 1f / MathF.Sqrt(variance + eps);
                meanOut[c] = mean;
                invStdOut[c] = invStd;
                for (int g = 0; g < groups; g++)
                {
                    float scale = invStd * gamma[g];
                    act[(g * channels) + c] = ((act[(g * channels) + c] - mean) * scale) + beta[g];
                }
            }
        }

        for (int c = 0; c < channels; c++)
        {
            float* dst = conv + (c * groups);
            for (int g = 0; g < groups; g++)
            {
                dst[g] = act[(g * channels) + c];
            }
        }
    }

    private unsafe void PackB()
    {
        int nr = _model.Nr;
        int tilesPer = _model.TilesPerSample;
        int depth = StandardV33Model.Depth;
        int channels = StandardV33Model.Channels;
        int kernel = StandardV33Model.Kernel;
        int groups = StandardV33Model.Groups;
        fixed (float* src0 = _conv)
        fixed (float* dst0 = _packedB)
        {
            for (int sample = 0; sample < _batch; sample++)
            {
                for (int tile = 0; tile < tilesPer; tile++)
                {
                    int oh0 = tile * nr;
                    int valid = Math.Min(nr, StandardV33Model.OutH - oh0);
                    float* dstTile = dst0 + (((sample * tilesPer) + tile) * depth * nr);
                    for (int ic = 0; ic < channels; ic++)
                    {
                        float* src = src0 + (((sample * channels) + ic) * groups) + oh0;
                        for (int kh = 0; kh < kernel; kh++)
                        {
                            float* dst = dstTile + (((ic * kernel) + kh) * nr);
                            if (valid > 0)
                            {
                                Unsafe.CopyBlockUnaligned(dst, src + kh, (uint)(valid * sizeof(float)));
                            }

                            for (int j = valid; j < nr; j++)
                            {
                                dst[j] = 0f;
                            }
                        }
                    }
                }
            }
        }
    }

    private void BeginFullGeometry()
    {
        _geomBatch = true;
        _geomTiles = _batch * _model.TilesPerSample;
        _geomPad = _batch * StandardV33Model.SampleStride;
        _geomColumns = StandardV33Model.OutH;
        _geomSample = 0;
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

    /// <summary>
    /// Padding groups are four copies of one embedding. After layer norm their convolution
    /// factors into a per-channel scale, so a pure 5-tap window is five multiply-adds per filter
    /// instead of the full 1280-wide product. Content windows still use the GEMM.
    /// </summary>
    private void ConvolveShortcutBatch()
    {
        Span<int> rangeStart = stackalloc int[StandardV33Model.OutH];
        Span<int> rangeEnd = stackalloc int[StandardV33Model.OutH];
        for (int sample = 0; sample < _batch; sample++)
        {
            if (_shortcut[sample] == 0)
            {
                if (_model.Portable)
                {
                    ConvolvePortableSample(sample, ReadOnlySpan<int>.Empty, activeCount: -1);
                }
                else
                {
                    PackSample(sample);
                    _geomBatch = false;
                    _geomSample = sample;
                    _geomTiles = _model.TilesPerSample;
                    _geomPad = StandardV33Model.SampleStride;
                    _geomColumns = StandardV33Model.OutH;
                    RunGeometry();
                }

                continue;
            }

            CollectPadWindows(sample, _activeOh, out int activeCount, rangeStart, rangeEnd, out int rangeCount);
            FillAnalytic(sample, rangeStart, rangeEnd, rangeCount);
            if (activeCount == 0)
            {
                _analytic.AsSpan(sample * StandardV33Model.Filters, StandardV33Model.Filters)
                    .CopyTo(_pooled.AsSpan(sample * StandardV33Model.Filters, StandardV33Model.Filters));
                continue;
            }

            if (_model.Portable)
            {
                ConvolvePortableSample(sample, _activeOh, activeCount);
            }
            else
            {
                int nr = _model.Nr;
                int nTiles = (activeCount + nr - 1) / nr;
                PackActive(sample, _activeOh, activeCount, nTiles);
                _geomBatch = false;
                _geomSample = sample;
                _geomTiles = nTiles;
                _geomPad = nTiles * nr;
                _geomColumns = activeCount;
                RunGeometry();
            }

            CombineAnalytic(sample);
        }
    }

    private bool IsPadShortcut(int sample, int paddingToken)
    {
        if (paddingToken != StandardV33Model.PadToken)
        {
            return false;
        }

        // A pure window is five fully padded groups. Fewer pad tokens than that run cannot qualify.
        int need = (StandardV33Model.MinPureWindows + StandardV33Model.Kernel - 1) * StandardV33Model.TokensPerGroup;
        int pads = 0;
        int origin = sample * StandardV33Model.Sequence;
        for (int i = 0; i < StandardV33Model.Sequence; i++)
        {
            if (_tokens[origin + i] == paddingToken)
            {
                pads++;
            }
        }

        if (pads < need)
        {
            return false;
        }

        return CountPureWindows(sample, paddingToken) >= StandardV33Model.MinPureWindows;
    }

    private int CountPureWindows(int sample, int paddingToken)
    {
        Span<int> active = stackalloc int[StandardV33Model.OutH];
        Span<int> rangeStart = stackalloc int[StandardV33Model.OutH];
        Span<int> rangeEnd = stackalloc int[StandardV33Model.OutH];
        CollectPadWindows(sample, active, out _, rangeStart, rangeEnd, out _, paddingToken, out int pure);
        return pure;
    }

    private void CollectPadWindows(int sample, Span<int> active, out int activeCount, Span<int> rangeStart, Span<int> rangeEnd, out int rangeCount) =>
        CollectPadWindows(sample, active, out activeCount, rangeStart, rangeEnd, out rangeCount, StandardV33Model.PadToken, out _);

    private void CollectPadWindows(
        int sample,
        Span<int> active,
        out int activeCount,
        Span<int> rangeStart,
        Span<int> rangeEnd,
        out int rangeCount,
        int paddingToken,
        out int pureWindows)
    {
        activeCount = 0;
        rangeCount = 0;
        pureWindows = 0;
        int groups = StandardV33Model.Groups;
        int kernel = StandardV33Model.Kernel;
        Span<byte> pure = stackalloc byte[groups];
        int origin = sample * StandardV33Model.Sequence;
        int step = StandardV33Model.TokensPerGroup;
        for (int g = 0; g < groups; g++)
        {
            int t = origin + (g * step);
            pure[g] = _tokens[t] == paddingToken
                && _tokens[t + 1] == paddingToken
                && _tokens[t + 2] == paddingToken
                && _tokens[t + 3] == paddingToken
                ? (byte)1
                : (byte)0;
        }

        int run = 0;
        for (int g = 0; g < kernel; g++)
        {
            run += pure[g];
        }

        int open = -1;
        for (int oh = 0; oh < StandardV33Model.OutH; oh++)
        {
            if (run == kernel)
            {
                pureWindows++;
                if (open < 0)
                {
                    open = oh;
                }
            }
            else
            {
                if (open >= 0)
                {
                    rangeStart[rangeCount] = open;
                    rangeEnd[rangeCount] = oh;
                    rangeCount++;
                    open = -1;
                }

                active[activeCount++] = oh;
            }

            if (oh + kernel < groups)
            {
                run += pure[oh + kernel] - pure[oh];
            }
        }

        if (open >= 0)
        {
            rangeStart[rangeCount] = open;
            rangeEnd[rangeCount] = StandardV33Model.OutH;
            rangeCount++;
        }
    }

    private unsafe void FillAnalytic(int sample, ReadOnlySpan<int> rangeStart, ReadOnlySpan<int> rangeEnd, int rangeCount)
    {
        BuildPadResponse(
            _savedMean.AsSpan(sample * StandardV33Model.Channels, StandardV33Model.Channels),
            _savedInv.AsSpan(sample * StandardV33Model.Channels, StandardV33Model.Channels));
        int filters = StandardV33Model.Filters;
        Span<float> max = _analytic.AsSpan(sample * filters, filters);
        max.Fill(float.NegativeInfinity);
        fixed (float* t0 = _padT)
        fixed (float* u0 = _model.WeightSum)
        fixed (float* gamma = _model.Ln0Gamma)
        fixed (float* beta = _model.Ln0Beta)
        fixed (float* acc = _padAcc)
        fixed (float* mx = max)
        {
            for (int r = 0; r < rangeCount; r++)
            {
                for (int oh = rangeStart[r]; oh < rangeEnd[r]; oh++)
                {
                    // out[f] = Σ_kh gamma[oh+kh] * T[kh,f] + beta[oh+kh] * U[kh,f]
                    // T and U already summed the 256 channels, so a padding window is five taps.
                    SetFma(acc, t0, gamma[oh], u0, beta[oh], filters);
                    for (int kh = 1; kh < StandardV33Model.Kernel; kh++)
                    {
                        int g = oh + kh;
                        Axpy(acc, t0 + (kh * filters), gamma[g], filters);
                        Axpy(acc, u0 + (kh * filters), beta[g], filters);
                    }

                    int f = 0;
                    if (Avx512F.IsSupported)
                    {
                        for (; f + 16 <= filters; f += 16)
                        {
                            Vector512<float> value = Avx512F.Max(Avx512F.LoadVector512(mx + f), Avx512F.LoadVector512(acc + f));
                            Avx512F.Store(mx + f, value);
                        }
                    }
                    else if (Avx2.IsSupported)
                    {
                        for (; f + 8 <= filters; f += 8)
                        {
                            Vector256<float> value = Avx.Max(Avx.LoadVector256(mx + f), Avx.LoadVector256(acc + f));
                            Avx.Store(mx + f, value);
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

    private unsafe void BuildPadResponse(ReadOnlySpan<float> mean, ReadOnlySpan<float> invStd)
    {
        int filters = StandardV33Model.Filters;
        int kernel = StandardV33Model.Kernel;
        _padT.AsSpan().Clear();
        fixed (float* t0 = _padT)
        fixed (float* w0 = _model.PadWeight)
        fixed (float* s0 = _model.PadAct)
        fixed (float* meanPtr = mean)
        fixed (float* invPtr = invStd)
        {
            for (int ic = 0; ic < StandardV33Model.Channels; ic++)
            {
                float alpha = (s0[ic] - meanPtr[ic]) * invPtr[ic];
                for (int kh = 0; kh < kernel; kh++)
                {
                    Axpy(t0 + (kh * filters), w0 + (((ic * kernel) + kh) * filters), alpha, filters);
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
                Vector512<float> product = Avx512F.Multiply(aFactor, Avx512F.LoadVector512(a + i));
                Vector512<float> value = Avx512F.FusedMultiplyAdd(bFactor, Avx512F.LoadVector512(b + i), product);
                Avx512F.Store(dst + i, value);
            }
        }
        else if (Avx2.IsSupported && Fma.IsSupported)
        {
            Vector256<float> aFactor = Vector256.Create(aScale);
            Vector256<float> bFactor = Vector256.Create(bScale);
            for (; i + 8 <= length; i += 8)
            {
                Vector256<float> product = Avx.Multiply(aFactor, Avx.LoadVector256(a + i));
                Vector256<float> value = Fma.MultiplyAdd(bFactor, Avx.LoadVector256(b + i), product);
                Avx.Store(dst + i, value);
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
                Vector512<float> value = Avx512F.FusedMultiplyAdd(factor, Avx512F.LoadVector512(src + i), Avx512F.LoadVector512(dst + i));
                Avx512F.Store(dst + i, value);
            }
        }
        else if (Avx2.IsSupported && Fma.IsSupported)
        {
            Vector256<float> factor = Vector256.Create(scale);
            for (; i + 8 <= length; i += 8)
            {
                Vector256<float> value = Fma.MultiplyAdd(factor, Avx.LoadVector256(src + i), Avx.LoadVector256(dst + i));
                Avx.Store(dst + i, value);
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

    private unsafe void PackSample(int sample)
    {
        int nr = _model.Nr;
        int tilesPer = _model.TilesPerSample;
        int depth = StandardV33Model.Depth;
        int channels = StandardV33Model.Channels;
        int kernel = StandardV33Model.Kernel;
        int groups = StandardV33Model.Groups;
        fixed (float* src0 = _conv)
        fixed (float* dst0 = _packedB)
        {
            for (int tile = 0; tile < tilesPer; tile++)
            {
                int oh0 = tile * nr;
                int valid = Math.Min(nr, StandardV33Model.OutH - oh0);
                float* dstTile = dst0 + (tile * depth * nr);
                for (int ic = 0; ic < channels; ic++)
                {
                    float* src = src0 + (((sample * channels) + ic) * groups) + oh0;
                    for (int kh = 0; kh < kernel; kh++)
                    {
                        float* dst = dstTile + (((ic * kernel) + kh) * nr);
                        if (valid > 0)
                        {
                            Unsafe.CopyBlockUnaligned(dst, src + kh, (uint)(valid * sizeof(float)));
                        }

                        for (int j = valid; j < nr; j++)
                        {
                            dst[j] = 0f;
                        }
                    }
                }
            }
        }
    }

    private unsafe void PackActive(int sample, int[] active, int activeCount, int nTiles)
    {
        int nr = _model.Nr;
        int depth = StandardV33Model.Depth;
        int channels = StandardV33Model.Channels;
        int kernel = StandardV33Model.Kernel;
        int groups = StandardV33Model.Groups;
        int packed = nTiles * depth * nr;
        _packedB.AsSpan(0, packed).Clear();
        fixed (float* src0 = _conv)
        fixed (float* dst0 = _packedB)
        {
            for (int j = 0; j < activeCount; j++)
            {
                int tile = j / nr;
                int lane = j - (tile * nr);
                int oh = active[j];
                for (int ic = 0; ic < channels; ic++)
                {
                    float* src = src0 + (((sample * channels) + ic) * groups) + oh;
                    for (int kh = 0; kh < kernel; kh++)
                    {
                        dst0[(((tile * depth) + (ic * kernel) + kh) * nr) + lane] = src[kh];
                    }
                }
            }
        }
    }

    private unsafe void ConvolvePortableSample(int sample, ReadOnlySpan<int> active, int activeCount)
    {
        bool full = activeCount < 0;
        int n = full ? StandardV33Model.OutH : activeCount;
        int depth = StandardV33Model.Depth;
        int channels = StandardV33Model.Channels;
        int kernel = StandardV33Model.Kernel;
        int groups = StandardV33Model.Groups;
        fixed (float* src0 = _conv)
        fixed (float* col0 = _packedB)
        {
            for (int ic = 0; ic < channels; ic++)
            {
                float* src = src0 + (((sample * channels) + ic) * groups);
                for (int kh = 0; kh < kernel; kh++)
                {
                    float* dst = col0 + (((ic * kernel) + kh) * n);
                    if (full)
                    {
                        Unsafe.CopyBlockUnaligned(dst, src + kh, (uint)(n * sizeof(float)));
                    }
                    else
                    {
                        for (int j = 0; j < n; j++)
                        {
                            dst[j] = src[active[j] + kh];
                        }
                    }
                }
            }
        }

        SimdGemm.Multiply(_model.ConvWeight, _packedB, _tiles, StandardV33Model.Filters, depth, n);
        for (int filter = 0; filter < StandardV33Model.Filters; filter++)
        {
            float max = float.NegativeInfinity;
            int row = filter * n;
            for (int j = 0; j < n; j++)
            {
                max = MathF.Max(max, _tiles[row + j]);
            }

            _pooled[(sample * StandardV33Model.Filters) + filter] = max;
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
    /// Streams each packed activation panel once per worker and reuses it across that worker's filters.
    /// K is blocked so the panel fits beside the weight slice in L1.
    /// </summary>
    private unsafe void ConvolveFilters(int tileStart, int tileEnd)
    {
        if (tileStart == tileEnd)
        {
            return;
        }

        const int Kc = 256;
        int nr = _model.Nr;
        int depth = StandardV33Model.Depth;
        int mr = StandardV33Model.Mr;
        int nPad = _geomPad;
        int nTiles = _geomTiles;
        fixed (float* a0 = _model.PackedA)
        fixed (float* b0 = _packedB)
        fixed (float* c0 = _convOut)
        fixed (float* pooled = _pooled)
        {
            for (int k0 = 0; k0 < depth; k0 += Kc)
            {
                int len = Math.Min(Kc, depth - k0);
                bool accumulate = k0 != 0;
                for (int col = 0; col < nTiles; col++)
                {
                    float* b = b0 + ((col * depth) + k0) * nr;
                    for (int filterTile = tileStart; filterTile < tileEnd; filterTile++)
                    {
                        float* a = a0 + ((filterTile * depth) + k0) * mr;
                        float* c = c0 + (filterTile * mr * nPad) + (col * nr);
                        if (nr == 32)
                        {
                            Kernel512_8x32(a, b, c, len, nPad, accumulate);
                        }
                        else
                        {
                            Kernel256_8x16(a, b, c, len, nPad, accumulate);
                        }
                    }
                }
            }

            for (int filterTile = tileStart; filterTile < tileEnd; filterTile++)
            {
                for (int row = 0; row < mr; row++)
                {
                    int filter = (filterTile * mr) + row;
                    float* src = c0 + (filter * nPad);
                    if (_geomBatch)
                    {
                        for (int sample = 0; sample < _batch; sample++)
                        {
                            pooled[(sample * StandardV33Model.Filters) + filter] = MaxPrefix(src + (sample * StandardV33Model.SampleStride), StandardV33Model.OutH);
                        }
                    }
                    else
                    {
                        pooled[(_geomSample * StandardV33Model.Filters) + filter] = MaxPrefix(src, _geomColumns);
                    }
                }
            }
        }
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
                acc = Avx512F.Max(acc, Avx512F.LoadVector512(row + j));
            }

            max = HorizontalMax(acc);
        }
        else if (Avx2.IsSupported)
        {
            Vector256<float> acc = Vector256.Create(float.NegativeInfinity);
            for (; j + 8 <= length; j += 8)
            {
                acc = Avx.Max(acc, Avx.LoadVector256(row + j));
            }

            max = HorizontalMax(acc);
        }

        for (; j < length; j++)
        {
            max = MathF.Max(max, row[j]);
        }

        return max;
    }

    private unsafe void ConvolvePortable()
    {
        int batch = _batch;
        int depth = StandardV33Model.Depth;
        int outH = StandardV33Model.OutH;
        int channels = StandardV33Model.Channels;
        int kernel = StandardV33Model.Kernel;
        int groups = StandardV33Model.Groups;
        int n = batch * outH;
        fixed (float* src0 = _conv)
        fixed (float* col0 = _packedB)
        {
            for (int sample = 0; sample < batch; sample++)
            {
                for (int ic = 0; ic < channels; ic++)
                {
                    float* src = src0 + (((sample * channels) + ic) * groups);
                    for (int kh = 0; kh < kernel; kh++)
                    {
                        float* dst = col0 + ((((ic * kernel) + kh) * n) + (sample * outH));
                        Unsafe.CopyBlockUnaligned(dst, src + kh, (uint)(outH * sizeof(float)));
                    }
                }
            }
        }

        SimdGemm.Multiply(_model.ConvWeight, _packedB, _tiles, StandardV33Model.Filters, depth, n);
        for (int sample = 0; sample < batch; sample++)
        {
            for (int filter = 0; filter < StandardV33Model.Filters; filter++)
            {
                float max = float.NegativeInfinity;
                int row = (filter * n) + (sample * outH);
                for (int oh = 0; oh < outH; oh++)
                {
                    max = MathF.Max(max, _tiles[row + oh]);
                }

                _pooled[(sample * StandardV33Model.Filters) + filter] = max;
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
                        Vector512<float> value = Avx512F.FusedMultiplyAdd(broadcast, Avx512F.LoadVector512(row + j), Avx512F.LoadVector512(acc + j));
                        Avx512F.Store(acc + j, value);
                    }
                }
                else if (Avx2.IsSupported && Fma.IsSupported)
                {
                    Vector256<float> broadcast = Vector256.Create(weight);
                    for (; j + 8 <= StandardV33Model.Classes; j += 8)
                    {
                        Vector256<float> value = Fma.MultiplyAdd(broadcast, Avx.LoadVector256(row + j), Avx.LoadVector256(acc + j));
                        Avx.Store(acc + j, value);
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

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel512_8x32(float* a, float* b, float* c, int k, int ldc, bool accumulate)
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
            r00 = Avx512F.LoadVector512(c);
            r01 = Avx512F.LoadVector512(c + 16);
            r10 = Avx512F.LoadVector512(c + ldc);
            r11 = Avx512F.LoadVector512(c + ldc + 16);
            r20 = Avx512F.LoadVector512(c + (2 * ldc));
            r21 = Avx512F.LoadVector512(c + (2 * ldc) + 16);
            r30 = Avx512F.LoadVector512(c + (3 * ldc));
            r31 = Avx512F.LoadVector512(c + (3 * ldc) + 16);
            r40 = Avx512F.LoadVector512(c + (4 * ldc));
            r41 = Avx512F.LoadVector512(c + (4 * ldc) + 16);
            r50 = Avx512F.LoadVector512(c + (5 * ldc));
            r51 = Avx512F.LoadVector512(c + (5 * ldc) + 16);
            r60 = Avx512F.LoadVector512(c + (6 * ldc));
            r61 = Avx512F.LoadVector512(c + (6 * ldc) + 16);
            r70 = Avx512F.LoadVector512(c + (7 * ldc));
            r71 = Avx512F.LoadVector512(c + (7 * ldc) + 16);
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

        for (int p = 0; p < k; p++)
        {
            Vector512<float> b0 = Avx512F.LoadVector512(b);
            Vector512<float> b1 = Avx512F.LoadVector512(b + 16);
            b += 32;
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

        Avx512F.Store(c, r00);
        Avx512F.Store(c + 16, r01);
        Avx512F.Store(c + ldc, r10);
        Avx512F.Store(c + ldc + 16, r11);
        Avx512F.Store(c + (2 * ldc), r20);
        Avx512F.Store(c + (2 * ldc) + 16, r21);
        Avx512F.Store(c + (3 * ldc), r30);
        Avx512F.Store(c + (3 * ldc) + 16, r31);
        Avx512F.Store(c + (4 * ldc), r40);
        Avx512F.Store(c + (4 * ldc) + 16, r41);
        Avx512F.Store(c + (5 * ldc), r50);
        Avx512F.Store(c + (5 * ldc) + 16, r51);
        Avx512F.Store(c + (6 * ldc), r60);
        Avx512F.Store(c + (6 * ldc) + 16, r61);
        Avx512F.Store(c + (7 * ldc), r70);
        Avx512F.Store(c + (7 * ldc) + 16, r71);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel256_8x16(float* a, float* b, float* c, int k, int ldc, bool accumulate)
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
            r00 = Avx.LoadVector256(c);
            r01 = Avx.LoadVector256(c + 8);
            r10 = Avx.LoadVector256(c + ldc);
            r11 = Avx.LoadVector256(c + ldc + 8);
            r20 = Avx.LoadVector256(c + (2 * ldc));
            r21 = Avx.LoadVector256(c + (2 * ldc) + 8);
            r30 = Avx.LoadVector256(c + (3 * ldc));
            r31 = Avx.LoadVector256(c + (3 * ldc) + 8);
            r40 = Avx.LoadVector256(c + (4 * ldc));
            r41 = Avx.LoadVector256(c + (4 * ldc) + 8);
            r50 = Avx.LoadVector256(c + (5 * ldc));
            r51 = Avx.LoadVector256(c + (5 * ldc) + 8);
            r60 = Avx.LoadVector256(c + (6 * ldc));
            r61 = Avx.LoadVector256(c + (6 * ldc) + 8);
            r70 = Avx.LoadVector256(c + (7 * ldc));
            r71 = Avx.LoadVector256(c + (7 * ldc) + 8);
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

        for (int p = 0; p < k; p++)
        {
            Vector256<float> b0 = Avx.LoadVector256(b);
            Vector256<float> b1 = Avx.LoadVector256(b + 8);
            b += 16;
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

        Avx.Store(c, r00);
        Avx.Store(c + 8, r01);
        Avx.Store(c + ldc, r10);
        Avx.Store(c + ldc + 8, r11);
        Avx.Store(c + (2 * ldc), r20);
        Avx.Store(c + (2 * ldc) + 8, r21);
        Avx.Store(c + (3 * ldc), r30);
        Avx.Store(c + (3 * ldc) + 8, r31);
        Avx.Store(c + (4 * ldc), r40);
        Avx.Store(c + (4 * ldc) + 8, r41);
        Avx.Store(c + (5 * ldc), r50);
        Avx.Store(c + (5 * ldc) + 8, r51);
        Avx.Store(c + (6 * ldc), r60);
        Avx.Store(c + (6 * ldc) + 8, r61);
        Avx.Store(c + (7 * ldc), r70);
        Avx.Store(c + (7 * ldc) + 8, r71);
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
}
