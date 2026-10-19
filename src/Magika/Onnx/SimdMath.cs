using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Magika.Onnx;

/// <summary>
/// Vector tanh and softmax for the activations that follow Magika's dense and convolution layers.
/// The exp reduction is the usual cephes argument split, evaluated with FMA.
/// On a finite softmax input it stays within a few ULPs of <see cref="MathF.Exp"/>; non-finite
/// logits take the scalar <see cref="MathF.Exp"/> path so infinities match that path exactly.
/// </summary>
internal static class SimdMath
{
    internal static void Tanh(ReadOnlySpan<float> source, Span<float> dest)
    {
        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Destination is shorter than the source.", nameof(dest));
        }

        if (Avx512F.IsSupported)
        {
            Tanh512(source, dest);
            return;
        }

        if (Avx2.IsSupported && Fma.IsSupported)
        {
            Tanh256(source, dest);
            return;
        }

        if (AdvSimd.Arm64.IsSupported)
        {
            Tanh128(source, dest);
            return;
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = MathF.Tanh(source[i]);
        }
    }

    /// <summary>
    /// Writes softmax(<paramref name="logits"/>) to <paramref name="dest"/>.
    /// Finite values use a vector cephes exp. The max is exact; exp is the same reduction as tanh.
    /// </summary>
    internal static void Softmax(Span<float> logits, Span<float> dest)
    {
        if (dest.Length < logits.Length)
        {
            throw new ArgumentException("Destination is shorter than the source.", nameof(dest));
        }

        if (logits.Length == 0)
        {
            return;
        }

        if (!AllFinite(logits))
        {
            SoftmaxScalar(logits, dest);
            return;
        }

        if (Avx512F.IsSupported)
        {
            Softmax512(logits, dest);
            return;
        }

        if (Avx2.IsSupported && Fma.IsSupported)
        {
            Softmax256(logits, dest);
            return;
        }

        if (AdvSimd.Arm64.IsSupported)
        {
            Softmax128(logits, dest);
            return;
        }

        SoftmaxScalar(logits, dest);
    }

    private static void SoftmaxScalar(ReadOnlySpan<float> logits, Span<float> dest)
    {
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
            dest[j] = exp;
            sum += exp;
        }

        float inv = 1f / sum;
        for (int j = 0; j < logits.Length; j++)
        {
            dest[j] *= inv;
        }
    }

    private static void Tanh512(ReadOnlySpan<float> source, Span<float> dest)
    {
        ref float src = ref MemoryMarshal.GetReference(source);
        ref float dst = ref MemoryMarshal.GetReference(dest);
        int i = 0;
        int length = source.Length;
        for (; i + 16 <= length; i += 16)
        {
            Vector512<float> value = Tanh(Vector512.LoadUnsafe(ref src, (nuint)i));
            value.StoreUnsafe(ref dst, (nuint)i);
        }

        for (; i < length; i++)
        {
            dest[i] = MathF.Tanh(source[i]);
        }
    }

    private static void Tanh256(ReadOnlySpan<float> source, Span<float> dest)
    {
        ref float src = ref MemoryMarshal.GetReference(source);
        ref float dst = ref MemoryMarshal.GetReference(dest);
        int i = 0;
        int length = source.Length;
        for (; i + 8 <= length; i += 8)
        {
            Vector256<float> value = Tanh(Vector256.LoadUnsafe(ref src, (nuint)i));
            value.StoreUnsafe(ref dst, (nuint)i);
        }

        for (; i < length; i++)
        {
            dest[i] = MathF.Tanh(source[i]);
        }
    }

    private static void Tanh128(ReadOnlySpan<float> source, Span<float> dest)
    {
        ref float src = ref MemoryMarshal.GetReference(source);
        ref float dst = ref MemoryMarshal.GetReference(dest);
        int i = 0;
        int length = source.Length;
        for (; i + 4 <= length; i += 4)
        {
            Vector128<float> value = Tanh(Vector128.LoadUnsafe(ref src, (nuint)i));
            value.StoreUnsafe(ref dst, (nuint)i);
        }

        for (; i < length; i++)
        {
            dest[i] = MathF.Tanh(source[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static Vector512<float> Tanh(Vector512<float> x)
    {
        Vector512<float> signMask = Vector512.Create(-0.0f);
        Vector512<float> sign = Vector512.BitwiseAnd(x, signMask);
        Vector512<float> abs = Vector512.AndNot(x, signMask);
        Vector512<float> exponent = Exp(abs * Vector512.Create(-2f));
        Vector512<float> one = Vector512<float>.One;
        Vector512<float> value = (one - exponent) / (one + exponent);
        Vector512<float> saturated = Avx512F.CompareGreaterThan(abs, Vector512.Create(9f));
        value = Vector512.ConditionalSelect(saturated, one, value);
        return Vector512.BitwiseOr(Vector512.AndNot(value, signMask), sign);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static Vector512<float> Exp(Vector512<float> x)
    {
        x = Avx512F.Min(Avx512F.Max(x, Vector512.Create(-87.3365478515625f)), Vector512.Create(88.3762626647949f));
        Vector512<float> fx = Avx512F.FusedMultiplyAdd(x, Vector512.Create(1.4426950408889634f), Vector512.Create(0.5f));
        fx = Vector512.Floor(fx);
        x = Avx512F.FusedMultiplyAdd(fx, Vector512.Create(-0.693359375f), x);
        x = Avx512F.FusedMultiplyAdd(fx, Vector512.Create(2.12194440e-4f), x);

        Vector512<float> z = x * x;
        Vector512<float> y = Vector512.Create(1.9875691500E-4f);
        y = Avx512F.FusedMultiplyAdd(y, x, Vector512.Create(1.3981999507E-3f));
        y = Avx512F.FusedMultiplyAdd(y, x, Vector512.Create(8.3334519073E-3f));
        y = Avx512F.FusedMultiplyAdd(y, x, Vector512.Create(4.1665795894E-2f));
        y = Avx512F.FusedMultiplyAdd(y, x, Vector512.Create(1.6666665459E-1f));
        y = Avx512F.FusedMultiplyAdd(y, x, Vector512.Create(5.0000001201E-1f));
        y = Avx512F.FusedMultiplyAdd(y, z, x);
        y += Vector512<float>.One;

        Vector512<int> exponent = Avx512F.ConvertToVector512Int32(fx);
        exponent = Avx512F.Add(exponent, Vector512.Create(127));
        exponent = Avx512F.ShiftLeftLogical(exponent, 23);
        return y * Vector512.As<int, float>(exponent);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static Vector256<float> Tanh(Vector256<float> x)
    {
        Vector256<float> signMask = Vector256.Create(-0.0f);
        Vector256<float> sign = Avx.And(x, signMask);
        Vector256<float> abs = Avx.AndNot(signMask, x);
        Vector256<float> exponent = Exp(Avx.Multiply(abs, Vector256.Create(-2f)));
        Vector256<float> one = Vector256<float>.One;
        Vector256<float> value = Avx.Divide(Avx.Subtract(one, exponent), Avx.Add(one, exponent));
        Vector256<float> saturated = Avx.CompareGreaterThan(abs, Vector256.Create(9f));
        value = Avx.BlendVariable(value, one, saturated);
        return Avx.Or(Avx.AndNot(signMask, value), sign);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static Vector256<float> Exp(Vector256<float> x)
    {
        x = Avx.Min(Avx.Max(x, Vector256.Create(-87.3365478515625f)), Vector256.Create(88.3762626647949f));
        Vector256<float> fx = Fma.MultiplyAdd(x, Vector256.Create(1.4426950408889634f), Vector256.Create(0.5f));
        fx = Avx.RoundToNegativeInfinity(fx);
        x = Fma.MultiplyAdd(fx, Vector256.Create(-0.693359375f), x);
        x = Fma.MultiplyAdd(fx, Vector256.Create(2.12194440e-4f), x);
        Vector256<float> z = Avx.Multiply(x, x);
        Vector256<float> y = Vector256.Create(1.9875691500E-4f);
        y = Fma.MultiplyAdd(y, x, Vector256.Create(1.3981999507E-3f));
        y = Fma.MultiplyAdd(y, x, Vector256.Create(8.3334519073E-3f));
        y = Fma.MultiplyAdd(y, x, Vector256.Create(4.1665795894E-2f));
        y = Fma.MultiplyAdd(y, x, Vector256.Create(1.6666665459E-1f));
        y = Fma.MultiplyAdd(y, x, Vector256.Create(5.0000001201E-1f));
        y = Fma.MultiplyAdd(y, z, x);
        y = Avx.Add(y, Vector256<float>.One);
        Vector256<int> exponent = Avx.ConvertToVector256Int32(fx);
        exponent = Avx2.Add(exponent, Vector256.Create(127));
        exponent = Avx2.ShiftLeftLogical(exponent, 23);
        return Avx.Multiply(y, exponent.As<int, float>());
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static Vector128<float> Tanh(Vector128<float> x)
    {
        Vector128<float> signMask = Vector128.Create(-0.0f);
        Vector128<float> sign = Vector128.BitwiseAnd(x, signMask);
        Vector128<float> abs = Vector128.AndNot(x, signMask);
        Vector128<float> exponent = Exp(AdvSimd.Multiply(abs, Vector128.Create(-2f)));
        Vector128<float> one = Vector128<float>.One;
        Vector128<float> value = AdvSimd.Arm64.Divide(AdvSimd.Subtract(one, exponent), AdvSimd.Add(one, exponent));
        Vector128<float> saturated = AdvSimd.CompareGreaterThan(abs, Vector128.Create(9f));
        value = Vector128.ConditionalSelect(saturated, one, value);
        return Vector128.BitwiseOr(Vector128.AndNot(value, signMask), sign);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static Vector128<float> Exp(Vector128<float> x)
    {
        x = AdvSimd.Min(AdvSimd.Max(x, Vector128.Create(-87.3365478515625f)), Vector128.Create(88.3762626647949f));
        Vector128<float> fx = AdvSimd.FusedMultiplyAdd(Vector128.Create(0.5f), x, Vector128.Create(1.4426950408889634f));
        fx = AdvSimd.Floor(fx);
        x = AdvSimd.FusedMultiplyAdd(x, fx, Vector128.Create(-0.693359375f));
        x = AdvSimd.FusedMultiplyAdd(x, fx, Vector128.Create(2.12194440e-4f));
        Vector128<float> z = AdvSimd.Multiply(x, x);
        Vector128<float> y = Vector128.Create(1.9875691500E-4f);
        y = AdvSimd.FusedMultiplyAdd(Vector128.Create(1.3981999507E-3f), y, x);
        y = AdvSimd.FusedMultiplyAdd(Vector128.Create(8.3334519073E-3f), y, x);
        y = AdvSimd.FusedMultiplyAdd(Vector128.Create(4.1665795894E-2f), y, x);
        y = AdvSimd.FusedMultiplyAdd(Vector128.Create(1.6666665459E-1f), y, x);
        y = AdvSimd.FusedMultiplyAdd(Vector128.Create(5.0000001201E-1f), y, x);
        y = AdvSimd.FusedMultiplyAdd(x, y, z);
        y = AdvSimd.Add(y, Vector128<float>.One);
        Vector128<int> exponent = AdvSimd.ConvertToInt32RoundToZero(fx);
        exponent = AdvSimd.Add(exponent, Vector128.Create(127));
        // Vector128<int> has no AdvSimd shift; the lane bits are an IEEE exponent either way.
        Vector128<uint> bits = AdvSimd.ShiftLeftLogical(exponent.As<int, uint>(), 23);
        return AdvSimd.Multiply(y, bits.As<uint, float>());
    }

    /// <summary>
    /// tanh-approximate GELU used by <c>standard_v3_3</c>, with the same multiply association as the graph:
    /// <c>x * (half * (1 + tanh(scale * (x + ((cubic * x) * x) * x))))</c>.
    /// </summary>
    internal static void Gelu(Span<float> data, float cubic, float scale, float one, float half)
    {
        if (Avx512F.IsSupported)
        {
            Gelu512(data, cubic, scale, one, half);
            return;
        }

        if (Avx2.IsSupported && Fma.IsSupported)
        {
            Gelu256(data, cubic, scale, one, half);
            return;
        }

        if (AdvSimd.Arm64.IsSupported)
        {
            Gelu128(data, cubic, scale, one, half);
            return;
        }

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = GeluScalar(data[i], cubic, scale, one, half);
        }
    }

    internal static float GeluScalar(float x, float cubic, float scale, float one, float half)
    {
        float cubicTerm = ((cubic * x) * x) * x;
        float inner = x + cubicTerm;
        float tanh = MathF.Tanh(scale * inner);
        return x * (half * (one + tanh));
    }

    private static void Gelu512(Span<float> data, float cubic, float scale, float one, float half)
    {
        Vector512<float> cubicV = Vector512.Create(cubic);
        Vector512<float> scaleV = Vector512.Create(scale);
        Vector512<float> oneV = Vector512.Create(one);
        Vector512<float> halfV = Vector512.Create(half);
        ref float start = ref MemoryMarshal.GetReference(data);
        int i = 0;
        for (; i + 16 <= data.Length; i += 16)
        {
            Vector512<float> x = Vector512.LoadUnsafe(ref start, (nuint)i);
            Vector512<float> cubicTerm = Avx512F.Multiply(Avx512F.Multiply(Avx512F.Multiply(cubicV, x), x), x);
            Vector512<float> inner = Avx512F.Add(x, cubicTerm);
            Vector512<float> tanh = Tanh(Avx512F.Multiply(scaleV, inner));
            Vector512<float> y = Avx512F.Multiply(x, Avx512F.Multiply(halfV, Avx512F.Add(oneV, tanh)));
            y.StoreUnsafe(ref start, (nuint)i);
        }

        for (; i < data.Length; i++)
        {
            data[i] = GeluScalar(data[i], cubic, scale, one, half);
        }
    }

    private static void Gelu256(Span<float> data, float cubic, float scale, float one, float half)
    {
        Vector256<float> cubicV = Vector256.Create(cubic);
        Vector256<float> scaleV = Vector256.Create(scale);
        Vector256<float> oneV = Vector256.Create(one);
        Vector256<float> halfV = Vector256.Create(half);
        ref float start = ref MemoryMarshal.GetReference(data);
        int i = 0;
        for (; i + 8 <= data.Length; i += 8)
        {
            Vector256<float> x = Vector256.LoadUnsafe(ref start, (nuint)i);
            Vector256<float> cubicTerm = Avx.Multiply(Avx.Multiply(Avx.Multiply(cubicV, x), x), x);
            Vector256<float> inner = Avx.Add(x, cubicTerm);
            Vector256<float> tanh = Tanh(Avx.Multiply(scaleV, inner));
            Vector256<float> y = Avx.Multiply(x, Avx.Multiply(halfV, Avx.Add(oneV, tanh)));
            y.StoreUnsafe(ref start, (nuint)i);
        }

        for (; i < data.Length; i++)
        {
            data[i] = GeluScalar(data[i], cubic, scale, one, half);
        }
    }

    private static void Gelu128(Span<float> data, float cubic, float scale, float one, float half)
    {
        Vector128<float> cubicV = Vector128.Create(cubic);
        Vector128<float> scaleV = Vector128.Create(scale);
        Vector128<float> oneV = Vector128.Create(one);
        Vector128<float> halfV = Vector128.Create(half);
        ref float start = ref MemoryMarshal.GetReference(data);
        int i = 0;
        for (; i + 4 <= data.Length; i += 4)
        {
            Vector128<float> x = Vector128.LoadUnsafe(ref start, (nuint)i);
            Vector128<float> cubicTerm = AdvSimd.Multiply(AdvSimd.Multiply(AdvSimd.Multiply(cubicV, x), x), x);
            Vector128<float> inner = AdvSimd.Add(x, cubicTerm);
            Vector128<float> tanh = Tanh(AdvSimd.Multiply(scaleV, inner));
            Vector128<float> y = AdvSimd.Multiply(x, AdvSimd.Multiply(halfV, AdvSimd.Add(oneV, tanh)));
            y.StoreUnsafe(ref start, (nuint)i);
        }

        for (; i < data.Length; i++)
        {
            data[i] = GeluScalar(data[i], cubic, scale, one, half);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Softmax512(Span<float> logits, Span<float> dest)
    {
        ref float src = ref MemoryMarshal.GetReference(logits);
        int length = logits.Length;
        int i = 0;
        Vector512<float> vmax = Vector512.Create(float.NegativeInfinity);
        for (; i + 16 <= length; i += 16)
        {
            vmax = Avx512F.Max(vmax, Vector512.LoadUnsafe(ref src, (nuint)i));
        }

        float max = FoldMax(vmax);
        for (; i < length; i++)
        {
            max = MathF.Max(max, logits[i]);
        }

        Vector512<float> shift = Vector512.Create(max);
        Vector512<float> sumV = Vector512<float>.Zero;
        i = 0;
        for (; i + 16 <= length; i += 16)
        {
            Vector512<float> exp = Exp(Avx512F.Subtract(Vector512.LoadUnsafe(ref src, (nuint)i), shift));
            exp.StoreUnsafe(ref src, (nuint)i);
            sumV = Avx512F.Add(sumV, exp);
        }

        float sum = FoldSum(sumV);
        for (; i < length; i++)
        {
            float exp = MathF.Exp(logits[i] - max);
            logits[i] = exp;
            sum += exp;
        }

        Scale(logits, dest, sum);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Softmax256(Span<float> logits, Span<float> dest)
    {
        ref float src = ref MemoryMarshal.GetReference(logits);
        int length = logits.Length;
        int i = 0;
        Vector256<float> vmax = Vector256.Create(float.NegativeInfinity);
        for (; i + 8 <= length; i += 8)
        {
            vmax = Avx.Max(vmax, Vector256.LoadUnsafe(ref src, (nuint)i));
        }

        float max = FoldMax(vmax);
        for (; i < length; i++)
        {
            max = MathF.Max(max, logits[i]);
        }

        Vector256<float> shift = Vector256.Create(max);
        Vector256<float> sumV = Vector256<float>.Zero;
        i = 0;
        for (; i + 8 <= length; i += 8)
        {
            Vector256<float> exp = Exp(Avx.Subtract(Vector256.LoadUnsafe(ref src, (nuint)i), shift));
            exp.StoreUnsafe(ref src, (nuint)i);
            sumV = Avx.Add(sumV, exp);
        }

        float sum = FoldSum(sumV);
        for (; i < length; i++)
        {
            float exp = MathF.Exp(logits[i] - max);
            logits[i] = exp;
            sum += exp;
        }

        Scale(logits, dest, sum);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Softmax128(Span<float> logits, Span<float> dest)
    {
        ref float src = ref MemoryMarshal.GetReference(logits);
        int length = logits.Length;
        int i = 0;
        Vector128<float> vmax = Vector128.Create(float.NegativeInfinity);
        for (; i + 4 <= length; i += 4)
        {
            vmax = AdvSimd.Max(vmax, Vector128.LoadUnsafe(ref src, (nuint)i));
        }

        float max = FoldMax(vmax);
        for (; i < length; i++)
        {
            max = MathF.Max(max, logits[i]);
        }

        Vector128<float> shift = Vector128.Create(max);
        Vector128<float> sumV = Vector128<float>.Zero;
        i = 0;
        for (; i + 4 <= length; i += 4)
        {
            Vector128<float> exp = Exp(AdvSimd.Subtract(Vector128.LoadUnsafe(ref src, (nuint)i), shift));
            exp.StoreUnsafe(ref src, (nuint)i);
            sumV = AdvSimd.Add(sumV, exp);
        }

        float sum = FoldSum(sumV);
        for (; i < length; i++)
        {
            float exp = MathF.Exp(logits[i] - max);
            logits[i] = exp;
            sum += exp;
        }

        Scale(logits, dest, sum);
    }

    private static void Scale(ReadOnlySpan<float> logits, Span<float> dest, float sum)
    {
        float inv = 1f / sum;
        ref float src = ref MemoryMarshal.GetReference(logits);
        ref float dst = ref MemoryMarshal.GetReference(dest);
        int length = logits.Length;
        int i = 0;
        if (Avx512F.IsSupported)
        {
            Vector512<float> scale = Vector512.Create(inv);
            for (; i + 16 <= length; i += 16)
            {
                Avx512F.Multiply(Vector512.LoadUnsafe(ref src, (nuint)i), scale).StoreUnsafe(ref dst, (nuint)i);
            }
        }
        else if (Avx2.IsSupported)
        {
            Vector256<float> scale = Vector256.Create(inv);
            for (; i + 8 <= length; i += 8)
            {
                Avx.Multiply(Vector256.LoadUnsafe(ref src, (nuint)i), scale).StoreUnsafe(ref dst, (nuint)i);
            }
        }
        else if (AdvSimd.Arm64.IsSupported)
        {
            Vector128<float> scale = Vector128.Create(inv);
            for (; i + 4 <= length; i += 4)
            {
                AdvSimd.Multiply(Vector128.LoadUnsafe(ref src, (nuint)i), scale).StoreUnsafe(ref dst, (nuint)i);
            }
        }

        for (; i < length; i++)
        {
            dest[i] = logits[i] * inv;
        }
    }

    private static float FoldMax(Vector512<float> value)
    {
        Vector256<float> folded = Avx.Max(value.GetLower(), value.GetUpper());
        return FoldMax(folded);
    }

    private static float FoldMax(Vector256<float> value)
    {
        Vector128<float> lower = value.GetLower();
        Vector128<float> upper = value.GetUpper();
        return FoldMax(Sse.Max(lower, upper));
    }

    private static float FoldMax(Vector128<float> value)
    {
        float max = value.GetElement(0);
        max = MathF.Max(max, value.GetElement(1));
        max = MathF.Max(max, value.GetElement(2));
        return MathF.Max(max, value.GetElement(3));
    }

    private static float FoldSum(Vector512<float> value)
    {
        Vector256<float> folded = Avx.Add(value.GetLower(), value.GetUpper());
        return FoldSum(folded);
    }

    private static float FoldSum(Vector256<float> value)
    {
        Vector128<float> folded = Sse.Add(value.GetLower(), value.GetUpper());
        return FoldSum(folded);
    }

    private static float FoldSum(Vector128<float> value) =>
        (value.GetElement(0) + value.GetElement(1)) + (value.GetElement(2) + value.GetElement(3));

    private static bool AllFinite(ReadOnlySpan<float> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
            {
                return false;
            }
        }

        return true;
    }
}
