using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Magika.Onnx;

/// <summary>
/// Vector tanh for the activations that follow Magika's dense and convolution layers.
/// The exp reduction is the usual cephes argument split, evaluated with FMA.
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

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = MathF.Tanh(source[i]);
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
}
