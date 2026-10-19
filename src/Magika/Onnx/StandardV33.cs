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

    /// <summary>
    /// Floats between channels in <c>_conv</c>. 512 groups plus room for a 32-wide load
    /// that starts on the last tile and shifts by the kernel. 528 is not a power of two,
    /// so consecutive channels do not collide in the same L1 sets.
    /// </summary>
    internal const int RowStride = 528;

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
        float[] fused = FuseEmbed(embed, embedBias, geluCubic, geluScale, geluOne, geluHalf);
        PackPadTables(convWeight, fused, out float[] padAct, out float[] padWeight, out float[] weightSum);
        return new StandardV33Model(
            fused,
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

    private static float[] FuseEmbed(float[] embed, float[] bias, float cubic, float scale, float one, float half)
    {
        float[] fused = GC.AllocateUninitializedArray<float>(Vocab * EmbedDim);
        for (int token = 0; token < Vocab; token++)
        {
            int offset = token * EmbedDim;
            for (int d = 0; d < EmbedDim; d++)
            {
                fused[offset + d] = embed[offset + d] + bias[d];
            }

            SimdMath.Gelu(fused.AsSpan(offset, EmbedDim), cubic, scale, one, half);
        }

        return fused;
    }

    private static void PackPadTables(
        float[] weight,
        float[] fusedEmbed,
        out float[] padAct,
        out float[] padWeight,
        out float[] weightSum)
    {
        padAct = new float[Channels];
        int pad = PadToken * EmbedDim;
        for (int c = 0; c < Channels; c++)
        {
            padAct[c] = fusedEmbed[pad + (c % EmbedDim)];
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
