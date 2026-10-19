using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Magika.Onnx;

/// <summary>
/// Row-major GEMM. The blocking and broadcast-FMA inner loop follow the same managed-kernel
/// approach as SimdPaddleOCR (handwritten AVX2/FMA, with a portable Vector fallback) but the
/// kernel itself is written for Magika's small dense and convolution shapes.
/// </summary>
internal static class SimdGemm
{
    internal static void Multiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        if (a.Length < (long)m * k || b.Length < (long)k * n || c.Length < (long)m * n)
        {
            throw new ArgumentException("GEMM buffer is smaller than m, k, n.");
        }

        c[..(m * n)].Clear();
        if (m == 0 || k == 0 || n == 0)
        {
            return;
        }

        if (Avx512F.IsSupported || Avx2.IsSupported)
        {
            MultiplyAvx(a, b, c, m, k, n);
            return;
        }

        MultiplyVector(a, b, c, m, k, n);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MultiplyAvx(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        const int ColumnBlock = 64;
        const int DepthBlock = 2048;
        int rowBlock = Avx512F.IsSupported ? 8 : 4;
        float[] panel = ArrayPoolRent(DepthBlock * ColumnBlock);
        try
        {
            for (int nc = 0; nc < n; nc += ColumnBlock)
            {
                int ncSize = Math.Min(ColumnBlock, n - nc);
                for (int kc = 0; kc < k; kc += DepthBlock)
                {
                    int kcSize = Math.Min(DepthBlock, k - kc);
                    PackB(b, k, n, kc, nc, kcSize, ncSize, panel);
                    int row = 0;
                    if (Avx512F.IsSupported)
                    {
                        for (; row + rowBlock <= m; row += rowBlock)
                        {
                            KernelAvx512Rows(a, c, panel, row, k, n, kc, nc, kcSize, ncSize);
                        }
                    }
                    else
                    {
                        for (; row + rowBlock <= m; row += rowBlock)
                        {
                            KernelAvx2Rows(a, c, panel, row, k, n, kc, nc, kcSize, ncSize);
                        }
                    }

                    for (; row < m; row++)
                    {
                        KernelRow(
                            a.Slice(row * k + kc, kcSize),
                            panel,
                            c.Slice(row * n + nc, ncSize),
                            kcSize,
                            ncSize);
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(panel);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void KernelAvx512Rows(
        ReadOnlySpan<float> a,
        Span<float> c,
        float[] panel,
        int row,
        int k,
        int n,
        int kc,
        int nc,
        int kcSize,
        int ncSize)
    {
        ref float bRef = ref panel[0];
        ref float a0 = ref RowA(a, row, k, kc, kcSize);
        ref float a1 = ref RowA(a, row + 1, k, kc, kcSize);
        ref float a2 = ref RowA(a, row + 2, k, kc, kcSize);
        ref float a3 = ref RowA(a, row + 3, k, kc, kcSize);
        ref float a4 = ref RowA(a, row + 4, k, kc, kcSize);
        ref float a5 = ref RowA(a, row + 5, k, kc, kcSize);
        ref float a6 = ref RowA(a, row + 6, k, kc, kcSize);
        ref float a7 = ref RowA(a, row + 7, k, kc, kcSize);
        ref float c0 = ref RowC(c, row, n, nc, ncSize);
        ref float c1 = ref RowC(c, row + 1, n, nc, ncSize);
        ref float c2 = ref RowC(c, row + 2, n, nc, ncSize);
        ref float c3 = ref RowC(c, row + 3, n, nc, ncSize);
        ref float c4 = ref RowC(c, row + 4, n, nc, ncSize);
        ref float c5 = ref RowC(c, row + 5, n, nc, ncSize);
        ref float c6 = ref RowC(c, row + 6, n, nc, ncSize);
        ref float c7 = ref RowC(c, row + 7, n, nc, ncSize);
        int j = 0;
        for (; j + 16 <= ncSize; j += 16)
        {
            Vector512<float> z0 = Vector512.LoadUnsafe(ref c0, (nuint)j);
            Vector512<float> z1 = Vector512.LoadUnsafe(ref c1, (nuint)j);
            Vector512<float> z2 = Vector512.LoadUnsafe(ref c2, (nuint)j);
            Vector512<float> z3 = Vector512.LoadUnsafe(ref c3, (nuint)j);
            Vector512<float> z4 = Vector512.LoadUnsafe(ref c4, (nuint)j);
            Vector512<float> z5 = Vector512.LoadUnsafe(ref c5, (nuint)j);
            Vector512<float> z6 = Vector512.LoadUnsafe(ref c6, (nuint)j);
            Vector512<float> z7 = Vector512.LoadUnsafe(ref c7, (nuint)j);
            for (int p = 0; p < kcSize; p++)
            {
                Vector512<float> bv = Vector512.LoadUnsafe(ref bRef, (nuint)(p * ncSize + j));
                z0 = Avx512F.FusedMultiplyAdd(Vector512.Create(Unsafe.Add(ref a0, p)), bv, z0);
                z1 = Avx512F.FusedMultiplyAdd(Vector512.Create(Unsafe.Add(ref a1, p)), bv, z1);
                z2 = Avx512F.FusedMultiplyAdd(Vector512.Create(Unsafe.Add(ref a2, p)), bv, z2);
                z3 = Avx512F.FusedMultiplyAdd(Vector512.Create(Unsafe.Add(ref a3, p)), bv, z3);
                z4 = Avx512F.FusedMultiplyAdd(Vector512.Create(Unsafe.Add(ref a4, p)), bv, z4);
                z5 = Avx512F.FusedMultiplyAdd(Vector512.Create(Unsafe.Add(ref a5, p)), bv, z5);
                z6 = Avx512F.FusedMultiplyAdd(Vector512.Create(Unsafe.Add(ref a6, p)), bv, z6);
                z7 = Avx512F.FusedMultiplyAdd(Vector512.Create(Unsafe.Add(ref a7, p)), bv, z7);
            }

            z0.StoreUnsafe(ref c0, (nuint)j);
            z1.StoreUnsafe(ref c1, (nuint)j);
            z2.StoreUnsafe(ref c2, (nuint)j);
            z3.StoreUnsafe(ref c3, (nuint)j);
            z4.StoreUnsafe(ref c4, (nuint)j);
            z5.StoreUnsafe(ref c5, (nuint)j);
            z6.StoreUnsafe(ref c6, (nuint)j);
            z7.StoreUnsafe(ref c7, (nuint)j);
        }

        TailColumns(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, ref bRef, ref c0, ref c1, ref c2, ref c3, ref c4, ref c5, ref c6, ref c7, kcSize, ncSize, j);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void KernelAvx2Rows(
        ReadOnlySpan<float> a,
        Span<float> c,
        float[] panel,
        int row,
        int k,
        int n,
        int kc,
        int nc,
        int kcSize,
        int ncSize)
    {
        ref float bRef = ref panel[0];
        ref float a0 = ref RowA(a, row, k, kc, kcSize);
        ref float a1 = ref RowA(a, row + 1, k, kc, kcSize);
        ref float a2 = ref RowA(a, row + 2, k, kc, kcSize);
        ref float a3 = ref RowA(a, row + 3, k, kc, kcSize);
        ref float c0 = ref RowC(c, row, n, nc, ncSize);
        ref float c1 = ref RowC(c, row + 1, n, nc, ncSize);
        ref float c2 = ref RowC(c, row + 2, n, nc, ncSize);
        ref float c3 = ref RowC(c, row + 3, n, nc, ncSize);
        bool fma = Fma.IsSupported;
        int j = 0;
        for (; j + 8 <= ncSize; j += 8)
        {
            Vector256<float> z0 = Vector256.LoadUnsafe(ref c0, (nuint)j);
            Vector256<float> z1 = Vector256.LoadUnsafe(ref c1, (nuint)j);
            Vector256<float> z2 = Vector256.LoadUnsafe(ref c2, (nuint)j);
            Vector256<float> z3 = Vector256.LoadUnsafe(ref c3, (nuint)j);
            for (int p = 0; p < kcSize; p++)
            {
                Vector256<float> bv = Vector256.LoadUnsafe(ref bRef, (nuint)(p * ncSize + j));
                z0 = Madd(Vector256.Create(Unsafe.Add(ref a0, p)), bv, z0, fma);
                z1 = Madd(Vector256.Create(Unsafe.Add(ref a1, p)), bv, z1, fma);
                z2 = Madd(Vector256.Create(Unsafe.Add(ref a2, p)), bv, z2, fma);
                z3 = Madd(Vector256.Create(Unsafe.Add(ref a3, p)), bv, z3, fma);
            }

            z0.StoreUnsafe(ref c0, (nuint)j);
            z1.StoreUnsafe(ref c1, (nuint)j);
            z2.StoreUnsafe(ref c2, (nuint)j);
            z3.StoreUnsafe(ref c3, (nuint)j);
        }

        for (; j < ncSize; j++)
        {
            float s0 = Unsafe.Add(ref c0, j);
            float s1 = Unsafe.Add(ref c1, j);
            float s2 = Unsafe.Add(ref c2, j);
            float s3 = Unsafe.Add(ref c3, j);
            for (int p = 0; p < kcSize; p++)
            {
                float bv = Unsafe.Add(ref bRef, p * ncSize + j);
                s0 += Unsafe.Add(ref a0, p) * bv;
                s1 += Unsafe.Add(ref a1, p) * bv;
                s2 += Unsafe.Add(ref a2, p) * bv;
                s3 += Unsafe.Add(ref a3, p) * bv;
            }

            Unsafe.Add(ref c0, j) = s0;
            Unsafe.Add(ref c1, j) = s1;
            Unsafe.Add(ref c2, j) = s2;
            Unsafe.Add(ref c3, j) = s3;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void KernelRow(ReadOnlySpan<float> aRow, float[] panel, Span<float> cRow, int kcSize, int ncSize)
    {
        ref float aRef = ref MemoryMarshal.GetReference(aRow);
        ref float bRef = ref panel[0];
        ref float cRef = ref MemoryMarshal.GetReference(cRow);
        int j = 0;
        if (Avx512F.IsSupported)
        {
            for (; j + 16 <= ncSize; j += 16)
            {
                Vector512<float> cv = Vector512.LoadUnsafe(ref cRef, (nuint)j);
                for (int p = 0; p < kcSize; p++)
                {
                    Vector512<float> bv = Vector512.LoadUnsafe(ref bRef, (nuint)(p * ncSize + j));
                    cv = Avx512F.FusedMultiplyAdd(Vector512.Create(Unsafe.Add(ref aRef, p)), bv, cv);
                }

                cv.StoreUnsafe(ref cRef, (nuint)j);
            }
        }

        if (Avx2.IsSupported)
        {
            bool fma = Fma.IsSupported;
            for (; j + 8 <= ncSize; j += 8)
            {
                Vector256<float> cv = Vector256.LoadUnsafe(ref cRef, (nuint)j);
                for (int p = 0; p < kcSize; p++)
                {
                    Vector256<float> bv = Vector256.LoadUnsafe(ref bRef, (nuint)(p * ncSize + j));
                    cv = Madd(Vector256.Create(Unsafe.Add(ref aRef, p)), bv, cv, fma);
                }

                cv.StoreUnsafe(ref cRef, (nuint)j);
            }
        }

        for (; j < ncSize; j++)
        {
            float acc = Unsafe.Add(ref cRef, j);
            for (int p = 0; p < kcSize; p++)
            {
                acc += Unsafe.Add(ref aRef, p) * Unsafe.Add(ref bRef, p * ncSize + j);
            }

            Unsafe.Add(ref cRef, j) = acc;
        }
    }

    private static void TailColumns(
        ref float a0, ref float a1, ref float a2, ref float a3, ref float a4, ref float a5, ref float a6, ref float a7,
        ref float bRef,
        ref float c0, ref float c1, ref float c2, ref float c3, ref float c4, ref float c5, ref float c6, ref float c7,
        int kcSize,
        int ncSize,
        int j)
    {
        if (Avx2.IsSupported)
        {
            bool fma = Fma.IsSupported;
            for (; j + 8 <= ncSize; j += 8)
            {
                Vector256<float> z0 = Vector256.LoadUnsafe(ref c0, (nuint)j);
                Vector256<float> z1 = Vector256.LoadUnsafe(ref c1, (nuint)j);
                Vector256<float> z2 = Vector256.LoadUnsafe(ref c2, (nuint)j);
                Vector256<float> z3 = Vector256.LoadUnsafe(ref c3, (nuint)j);
                Vector256<float> z4 = Vector256.LoadUnsafe(ref c4, (nuint)j);
                Vector256<float> z5 = Vector256.LoadUnsafe(ref c5, (nuint)j);
                Vector256<float> z6 = Vector256.LoadUnsafe(ref c6, (nuint)j);
                Vector256<float> z7 = Vector256.LoadUnsafe(ref c7, (nuint)j);
                for (int p = 0; p < kcSize; p++)
                {
                    Vector256<float> bv = Vector256.LoadUnsafe(ref bRef, (nuint)(p * ncSize + j));
                    z0 = Madd(Vector256.Create(Unsafe.Add(ref a0, p)), bv, z0, fma);
                    z1 = Madd(Vector256.Create(Unsafe.Add(ref a1, p)), bv, z1, fma);
                    z2 = Madd(Vector256.Create(Unsafe.Add(ref a2, p)), bv, z2, fma);
                    z3 = Madd(Vector256.Create(Unsafe.Add(ref a3, p)), bv, z3, fma);
                    z4 = Madd(Vector256.Create(Unsafe.Add(ref a4, p)), bv, z4, fma);
                    z5 = Madd(Vector256.Create(Unsafe.Add(ref a5, p)), bv, z5, fma);
                    z6 = Madd(Vector256.Create(Unsafe.Add(ref a6, p)), bv, z6, fma);
                    z7 = Madd(Vector256.Create(Unsafe.Add(ref a7, p)), bv, z7, fma);
                }

                z0.StoreUnsafe(ref c0, (nuint)j);
                z1.StoreUnsafe(ref c1, (nuint)j);
                z2.StoreUnsafe(ref c2, (nuint)j);
                z3.StoreUnsafe(ref c3, (nuint)j);
                z4.StoreUnsafe(ref c4, (nuint)j);
                z5.StoreUnsafe(ref c5, (nuint)j);
                z6.StoreUnsafe(ref c6, (nuint)j);
                z7.StoreUnsafe(ref c7, (nuint)j);
            }
        }

        for (; j < ncSize; j++)
        {
            float s0 = Unsafe.Add(ref c0, j);
            float s1 = Unsafe.Add(ref c1, j);
            float s2 = Unsafe.Add(ref c2, j);
            float s3 = Unsafe.Add(ref c3, j);
            float s4 = Unsafe.Add(ref c4, j);
            float s5 = Unsafe.Add(ref c5, j);
            float s6 = Unsafe.Add(ref c6, j);
            float s7 = Unsafe.Add(ref c7, j);
            for (int p = 0; p < kcSize; p++)
            {
                float bv = Unsafe.Add(ref bRef, p * ncSize + j);
                s0 += Unsafe.Add(ref a0, p) * bv;
                s1 += Unsafe.Add(ref a1, p) * bv;
                s2 += Unsafe.Add(ref a2, p) * bv;
                s3 += Unsafe.Add(ref a3, p) * bv;
                s4 += Unsafe.Add(ref a4, p) * bv;
                s5 += Unsafe.Add(ref a5, p) * bv;
                s6 += Unsafe.Add(ref a6, p) * bv;
                s7 += Unsafe.Add(ref a7, p) * bv;
            }

            Unsafe.Add(ref c0, j) = s0;
            Unsafe.Add(ref c1, j) = s1;
            Unsafe.Add(ref c2, j) = s2;
            Unsafe.Add(ref c3, j) = s3;
            Unsafe.Add(ref c4, j) = s4;
            Unsafe.Add(ref c5, j) = s5;
            Unsafe.Add(ref c6, j) = s6;
            Unsafe.Add(ref c7, j) = s7;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Madd(Vector256<float> a, Vector256<float> b, Vector256<float> c, bool fma) =>
        fma ? Fma.MultiplyAdd(a, b, c) : Avx.Add(c, Avx.Multiply(a, b));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ref float RowA(ReadOnlySpan<float> a, int row, int k, int kc, int kcSize) =>
        ref MemoryMarshal.GetReference(a.Slice(row * k + kc, kcSize));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ref float RowC(Span<float> c, int row, int n, int nc, int ncSize) =>
        ref MemoryMarshal.GetReference(c.Slice(row * n + nc, ncSize));

    private static void PackB(ReadOnlySpan<float> b, int k, int n, int kc, int nc, int kcSize, int ncSize, float[] panel)
    {
        int dest = 0;
        for (int p = 0; p < kcSize; p++)
        {
            int src = (kc + p) * n + nc;
            b.Slice(src, ncSize).CopyTo(panel.AsSpan(dest, ncSize));
            dest += ncSize;
        }

        _ = k;
    }

    private static void MultiplyVector(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        int width = Vector<float>.Count;
        for (int i = 0; i < m; i++)
        {
            int cRow = i * n;
            int aRow = i * k;
            for (int p = 0; p < k; p++)
            {
                float av = a[aRow + p];
                var aVec = new Vector<float>(av);
                int bRow = p * n;
                int j = 0;
                for (; j + width <= n; j += width)
                {
                    var cv = new Vector<float>(c.Slice(cRow + j, width));
                    var bv = new Vector<float>(b.Slice(bRow + j, width));
                    (cv + aVec * bv).CopyTo(c.Slice(cRow + j, width));
                }

                for (; j < n; j++)
                {
                    c[cRow + j] += av * b[bRow + j];
                }
            }
        }
    }

    private static float[] ArrayPoolRent(int length) => System.Buffers.ArrayPool<float>.Shared.Rent(length);
}
