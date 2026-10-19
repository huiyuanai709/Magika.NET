using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Magika.Onnx;

/// <summary>
/// Row-major GEMM for Magika's convolution and dense shapes.
/// Large products pack B into contiguous SIMD panels and run a broadcast-FMA microkernel,
/// the same blocked managed-kernel approach SimdPaddleOCR uses, written for these shapes.
/// </summary>
internal static class SimdGemm
{
    private const int LargeWork = 200_000;

    /// <summary>
    /// Caps GEMM parallelism on the calling thread. <c>0</c> uses every processor.
    /// Read before work is split, so pool threads do not consult it.
    /// </summary>
    [ThreadStatic]
    internal static int ThreadBudget;

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

        long work = (long)m * k * n;
        if (work >= LargeWork && m >= 8 && n >= 8)
        {
            if (Avx512F.IsSupported)
            {
                MultiplyBlockedAvx512(a, b, c, m, k, n);
                return;
            }

            if (Avx2.IsSupported)
            {
                MultiplyBlockedAvx2(a, b, c, m, k, n);
                return;
            }

            MultiplyVectorParallel(a, b, c, m, k, n);
            return;
        }

        if (Avx512F.IsSupported || Avx2.IsSupported)
        {
            MultiplyAvx(a, b, c, m, k, n);
            return;
        }

        MultiplyVector(a, b, c, m, k, n);
    }

    private static unsafe void MultiplyBlockedAvx512(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        const int Nr = 32;
        int full = n / Nr;
        int tail = n - (full * Nr);
        int packed = checked(k * ((full * Nr) + tail));
        float[] panel = ArrayPoolRent(packed + 16);
        try
        {
            fixed (float* aPtr = a)
            fixed (float* bSrc = b)
            fixed (float* raw = panel)
            fixed (float* cPtr = c)
            {
                float* bPtr = Align64(raw);
                PackB(bSrc, bPtr, k, n, full, Nr, tail);
                RunRows(aPtr, bPtr, cPtr, m, k, n, full, Nr, tail, KernelKind.Avx512);
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(panel);
        }
    }

    private static unsafe void MultiplyBlockedAvx2(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        const int Nr = 16;
        int full = n / Nr;
        int tail = n - (full * Nr);
        int packed = checked(k * ((full * Nr) + tail));
        float[] panel = ArrayPoolRent(packed + 16);
        try
        {
            fixed (float* aPtr = a)
            fixed (float* bSrc = b)
            fixed (float* raw = panel)
            fixed (float* cPtr = c)
            {
                float* bPtr = Align64(raw);
                PackB(bSrc, bPtr, k, n, full, Nr, tail);
                RunRows(aPtr, bPtr, cPtr, m, k, n, full, Nr, tail, KernelKind.Avx2);
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(panel);
        }
    }

    private static unsafe void MultiplyVectorParallel(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int m, int k, int n)
    {
        fixed (float* aPtr = a)
        fixed (float* bPtr = b)
        fixed (float* cPtr = c)
        {
            int threads = ChooseThreads(m);
            if (threads == 1)
            {
                MultiplyVectorRange(aPtr, bPtr, cPtr, 0, m, k, n);
                return;
            }

            var job = new VectorJob
            {
                A = aPtr,
                B = bPtr,
                C = cPtr,
                M = m,
                K = k,
                N = n,
                Threads = threads,
            };
            Parallel.For(0, threads, job.Invoke);
        }
    }

    private static unsafe void RunRows(float* a, float* b, float* c, int m, int k, int n, int full, int nr, int tail, KernelKind kind)
    {
        const int Mr = 8;
        int tiles = m / Mr;
        int threads = ChooseThreads(tiles);
        if (threads == 1)
        {
            ComputeTiles(a, b, c, 0, tiles, m, k, n, full, nr, tail, kind);
            return;
        }

        var job = new TileJob
        {
            A = a,
            B = b,
            C = c,
            Tiles = tiles,
            M = m,
            K = k,
            N = n,
            Full = full,
            Nr = nr,
            Tail = tail,
            Kind = kind,
            Threads = threads,
        };
        Parallel.For(0, threads, job.Invoke);
    }

    private static int ChooseThreads(int units)
    {
        int threads = Environment.ProcessorCount;
        if (ThreadBudget > 0)
        {
            threads = Math.Min(threads, ThreadBudget);
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

    private static unsafe void PackB(float* b, float* packed, int k, int n, int full, int nr, int tail)
    {
        for (int tile = 0; tile < full; tile++)
        {
            float* dst = packed + ((long)tile * k * nr);
            int col = tile * nr;
            for (int p = 0; p < k; p++)
            {
                Unsafe.CopyBlockUnaligned(dst + ((long)p * nr), b + ((long)p * n) + col, (uint)(nr * sizeof(float)));
            }
        }

        if (tail == 0)
        {
            return;
        }

        float* tailDst = packed + ((long)full * k * nr);
        int tailCol = full * nr;
        for (int p = 0; p < k; p++)
        {
            Unsafe.CopyBlockUnaligned(tailDst + ((long)p * tail), b + ((long)p * n) + tailCol, (uint)(tail * sizeof(float)));
        }
    }

    private enum KernelKind
    {
        Avx512,
        Avx2,
    }

    private sealed unsafe class TileJob
    {
        public float* A;
        public float* B;
        public float* C;
        public int Tiles;
        public int M;
        public int K;
        public int N;
        public int Full;
        public int Nr;
        public int Tail;
        public KernelKind Kind;
        public int Threads;

        public void Invoke(int id)
        {
            int start = (int)((long)id * Tiles / Threads);
            int end = (int)((long)(id + 1) * Tiles / Threads);
            ComputeTiles(A, B, C, start, end, id == Threads - 1 ? M : end * 8, K, N, Full, Nr, Tail, Kind);
        }
    }

    private static unsafe void ComputeTiles(
        float* a,
        float* b,
        float* c,
        int tileStart,
        int tileEnd,
        int rowLimit,
        int k,
        int n,
        int full,
        int nr,
        int tail,
        KernelKind kind)
    {
        const int Mr = 8;
        for (int tile = 0; tile < full; tile++)
        {
            float* panel = b + ((long)tile * k * nr);
            int col = tile * nr;
            for (int t = tileStart; t < tileEnd; t++)
            {
                int row = t * Mr;
                float* ar = a + ((long)row * k);
                float* cr = c + ((long)row * n) + col;
                if (kind == KernelKind.Avx512)
                {
                    Kernel512_8x32(ar, k, panel, cr, n);
                }
                else
                {
                    Kernel256_8x16(ar, k, panel, cr, n);
                }
            }
        }

        if (tail > 0)
        {
            float* panel = b + ((long)full * k * nr);
            int col = full * nr;
            for (int t = tileStart; t < tileEnd; t++)
            {
                int row = t * Mr;
                ApplyTail(a + ((long)row * k), k, panel, tail, c + ((long)row * n) + col, n, kind);
            }
        }

        int covered = tileEnd * Mr;
        if (covered < rowLimit)
        {
            MultiplyScalarRows(a, b, c, covered, rowLimit, k, n, full, nr, tail);
        }
    }

    private static unsafe void ApplyTail(float* a, int k, float* panel, int tail, float* c, int n, KernelKind kind)
    {
        int done = 0;
        if (kind == KernelKind.Avx512 && tail - done >= 16)
        {
            Kernel512_8x16(a, k, panel + done, tail, c + done, n);
            done += 16;
        }

        if (tail - done >= 8 && (kind == KernelKind.Avx2 || Avx2.IsSupported))
        {
            Kernel256_8x8(a, k, panel + done, tail, c + done, n);
            done += 8;
        }

        if (done < tail)
        {
            KernelScalar8(a, k, panel + done, tail, c + done, n, tail - done);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel512_8x32(float* a, int k, float* b, float* c, int ldc)
    {
        float* a0 = a;
        float* a1 = a + k;
        float* a2 = a + (2 * k);
        float* a3 = a + (3 * k);
        float* a4 = a + (4 * k);
        float* a5 = a + (5 * k);
        float* a6 = a + (6 * k);
        float* a7 = a + (7 * k);
        Vector512<float> z00 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c));
        Vector512<float> z01 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c), 16);
        Vector512<float> z10 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + ldc));
        Vector512<float> z11 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + ldc), 16);
        Vector512<float> z20 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
        Vector512<float> z21 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)), 16);
        Vector512<float> z30 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
        Vector512<float> z31 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)), 16);
        Vector512<float> z40 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
        Vector512<float> z41 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)), 16);
        Vector512<float> z50 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
        Vector512<float> z51 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)), 16);
        Vector512<float> z60 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
        Vector512<float> z61 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)), 16);
        Vector512<float> z70 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
        Vector512<float> z71 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)), 16);
        float* bp = b;
        for (int p = 0; p < k; p++)
        {
            Fma512x8x32(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref a6, ref a7, ref bp,
                ref z00, ref z01, ref z10, ref z11, ref z20, ref z21, ref z30, ref z31,
                ref z40, ref z41, ref z50, ref z51, ref z60, ref z61, ref z70, ref z71);
        }

        Store512Pair(c, z00, z01);
        Store512Pair(c + ldc, z10, z11);
        Store512Pair(c + (2 * ldc), z20, z21);
        Store512Pair(c + (3 * ldc), z30, z31);
        Store512Pair(c + (4 * ldc), z40, z41);
        Store512Pair(c + (5 * ldc), z50, z51);
        Store512Pair(c + (6 * ldc), z60, z61);
        Store512Pair(c + (7 * ldc), z70, z71);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Fma512x8x32(
        ref float* a0, ref float* a1, ref float* a2, ref float* a3, ref float* a4, ref float* a5, ref float* a6, ref float* a7, ref float* b,
        ref Vector512<float> z00, ref Vector512<float> z01, ref Vector512<float> z10, ref Vector512<float> z11,
        ref Vector512<float> z20, ref Vector512<float> z21, ref Vector512<float> z30, ref Vector512<float> z31,
        ref Vector512<float> z40, ref Vector512<float> z41, ref Vector512<float> z50, ref Vector512<float> z51,
        ref Vector512<float> z60, ref Vector512<float> z61, ref Vector512<float> z70, ref Vector512<float> z71)
    {
        Vector512<float> b0 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(b));
        Vector512<float> b1 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(b), 16);
        z00 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a0), b0, z00);
        z01 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a0), b1, z01);
        z10 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a1), b0, z10);
        z11 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a1), b1, z11);
        z20 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a2), b0, z20);
        z21 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a2), b1, z21);
        z30 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a3), b0, z30);
        z31 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a3), b1, z31);
        z40 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a4), b0, z40);
        z41 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a4), b1, z41);
        z50 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a5), b0, z50);
        z51 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a5), b1, z51);
        z60 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a6), b0, z60);
        z61 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a6), b1, z61);
        z70 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a7), b0, z70);
        z71 = Avx512F.FusedMultiplyAdd(Vector512.Create(*a7), b1, z71);
        a0++;
        a1++;
        a2++;
        a3++;
        a4++;
        a5++;
        a6++;
        a7++;
        b += 32;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store512Pair(float* c, Vector512<float> low, Vector512<float> high)
    {
        low.StoreUnsafe(ref Unsafe.AsRef<float>(c));
        high.StoreUnsafe(ref Unsafe.AsRef<float>(c), 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel512_8x16(float* a, int k, float* b, int stride, float* c, int ldc)
    {
        Vector512<float> z0 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c));
        Vector512<float> z1 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + ldc));
        Vector512<float> z2 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
        Vector512<float> z3 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
        Vector512<float> z4 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
        Vector512<float> z5 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
        Vector512<float> z6 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
        Vector512<float> z7 = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
        for (int p = 0; p < k; p++)
        {
            Vector512<float> bv = Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(b + ((long)p * stride)));
            z0 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[p]), bv, z0);
            z1 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[k + p]), bv, z1);
            z2 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[(2 * k) + p]), bv, z2);
            z3 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[(3 * k) + p]), bv, z3);
            z4 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[(4 * k) + p]), bv, z4);
            z5 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[(5 * k) + p]), bv, z5);
            z6 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[(6 * k) + p]), bv, z6);
            z7 = Avx512F.FusedMultiplyAdd(Vector512.Create(a[(7 * k) + p]), bv, z7);
        }

        z0.StoreUnsafe(ref Unsafe.AsRef<float>(c));
        z1.StoreUnsafe(ref Unsafe.AsRef<float>(c + ldc));
        z2.StoreUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
        z3.StoreUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
        z4.StoreUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
        z5.StoreUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
        z6.StoreUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
        z7.StoreUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel256_8x16(float* a, int k, float* b, float* c, int ldc)
    {
        bool fma = Fma.IsSupported;
        Vector256<float> z00 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c));
        Vector256<float> z01 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c), 8);
        Vector256<float> z10 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + ldc));
        Vector256<float> z11 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + ldc), 8);
        Vector256<float> z20 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
        Vector256<float> z21 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)), 8);
        Vector256<float> z30 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
        Vector256<float> z31 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)), 8);
        Vector256<float> z40 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
        Vector256<float> z41 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)), 8);
        Vector256<float> z50 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
        Vector256<float> z51 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)), 8);
        Vector256<float> z60 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
        Vector256<float> z61 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)), 8);
        Vector256<float> z70 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
        Vector256<float> z71 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)), 8);
        for (int p = 0; p < k; p++)
        {
            Vector256<float> b0 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(b + ((long)p * 16)));
            Vector256<float> b1 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(b + ((long)p * 16)), 8);
            z00 = Madd(Vector256.Create(a[p]), b0, z00, fma);
            z01 = Madd(Vector256.Create(a[p]), b1, z01, fma);
            z10 = Madd(Vector256.Create(a[k + p]), b0, z10, fma);
            z11 = Madd(Vector256.Create(a[k + p]), b1, z11, fma);
            z20 = Madd(Vector256.Create(a[(2 * k) + p]), b0, z20, fma);
            z21 = Madd(Vector256.Create(a[(2 * k) + p]), b1, z21, fma);
            z30 = Madd(Vector256.Create(a[(3 * k) + p]), b0, z30, fma);
            z31 = Madd(Vector256.Create(a[(3 * k) + p]), b1, z31, fma);
            z40 = Madd(Vector256.Create(a[(4 * k) + p]), b0, z40, fma);
            z41 = Madd(Vector256.Create(a[(4 * k) + p]), b1, z41, fma);
            z50 = Madd(Vector256.Create(a[(5 * k) + p]), b0, z50, fma);
            z51 = Madd(Vector256.Create(a[(5 * k) + p]), b1, z51, fma);
            z60 = Madd(Vector256.Create(a[(6 * k) + p]), b0, z60, fma);
            z61 = Madd(Vector256.Create(a[(6 * k) + p]), b1, z61, fma);
            z70 = Madd(Vector256.Create(a[(7 * k) + p]), b0, z70, fma);
            z71 = Madd(Vector256.Create(a[(7 * k) + p]), b1, z71, fma);
        }

        z00.StoreUnsafe(ref Unsafe.AsRef<float>(c));
        z01.StoreUnsafe(ref Unsafe.AsRef<float>(c), 8);
        z10.StoreUnsafe(ref Unsafe.AsRef<float>(c + ldc));
        z11.StoreUnsafe(ref Unsafe.AsRef<float>(c + ldc), 8);
        z20.StoreUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
        z21.StoreUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)), 8);
        z30.StoreUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
        z31.StoreUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)), 8);
        z40.StoreUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
        z41.StoreUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)), 8);
        z50.StoreUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
        z51.StoreUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)), 8);
        z60.StoreUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
        z61.StoreUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)), 8);
        z70.StoreUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
        z71.StoreUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)), 8);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel256_8x8(float* a, int k, float* b, int stride, float* c, int ldc)
    {
        bool fma = Fma.IsSupported;
        Vector256<float> z0 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c));
        Vector256<float> z1 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + ldc));
        Vector256<float> z2 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
        Vector256<float> z3 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
        Vector256<float> z4 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
        Vector256<float> z5 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
        Vector256<float> z6 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
        Vector256<float> z7 = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
        for (int p = 0; p < k; p++)
        {
            Vector256<float> bv = Vector256.LoadUnsafe(ref Unsafe.AsRef<float>(b + ((long)p * stride)));
            z0 = Madd(Vector256.Create(a[p]), bv, z0, fma);
            z1 = Madd(Vector256.Create(a[k + p]), bv, z1, fma);
            z2 = Madd(Vector256.Create(a[(2 * k) + p]), bv, z2, fma);
            z3 = Madd(Vector256.Create(a[(3 * k) + p]), bv, z3, fma);
            z4 = Madd(Vector256.Create(a[(4 * k) + p]), bv, z4, fma);
            z5 = Madd(Vector256.Create(a[(5 * k) + p]), bv, z5, fma);
            z6 = Madd(Vector256.Create(a[(6 * k) + p]), bv, z6, fma);
            z7 = Madd(Vector256.Create(a[(7 * k) + p]), bv, z7, fma);
        }

        z0.StoreUnsafe(ref Unsafe.AsRef<float>(c));
        z1.StoreUnsafe(ref Unsafe.AsRef<float>(c + ldc));
        z2.StoreUnsafe(ref Unsafe.AsRef<float>(c + (2 * ldc)));
        z3.StoreUnsafe(ref Unsafe.AsRef<float>(c + (3 * ldc)));
        z4.StoreUnsafe(ref Unsafe.AsRef<float>(c + (4 * ldc)));
        z5.StoreUnsafe(ref Unsafe.AsRef<float>(c + (5 * ldc)));
        z6.StoreUnsafe(ref Unsafe.AsRef<float>(c + (6 * ldc)));
        z7.StoreUnsafe(ref Unsafe.AsRef<float>(c + (7 * ldc)));
    }

    private static unsafe void KernelScalar8(float* a, int k, float* b, int stride, float* c, int ldc, int width)
    {
        for (int row = 0; row < 8; row++)
        {
            float* ar = a + ((long)row * k);
            float* cr = c + ((long)row * ldc);
            for (int j = 0; j < width; j++)
            {
                float sum = cr[j];
                for (int p = 0; p < k; p++)
                {
                    sum += ar[p] * b[((long)p * stride) + j];
                }

                cr[j] = sum;
            }
        }
    }

    /// <summary>
    /// Leftover rows after the 8-row kernels. B is the packed panel, not the original matrix.
    /// </summary>
    private static unsafe void MultiplyScalarRows(float* a, float* packedB, float* c, int row0, int row1, int k, int n, int full, int nr, int tail)
    {
        for (int row = row0; row < row1; row++)
        {
            float* ar = a + ((long)row * k);
            for (int tile = 0; tile < full; tile++)
            {
                float* panel = packedB + ((long)tile * k * nr);
                int col = tile * nr;
                for (int j = 0; j < nr; j++)
                {
                    float sum = c[((long)row * n) + col + j];
                    for (int p = 0; p < k; p++)
                    {
                        sum += ar[p] * panel[((long)p * nr) + j];
                    }

                    c[((long)row * n) + col + j] = sum;
                }
            }

            if (tail > 0)
            {
                float* panel = packedB + ((long)full * k * nr);
                int col = full * nr;
                for (int j = 0; j < tail; j++)
                {
                    float sum = c[((long)row * n) + col + j];
                    for (int p = 0; p < k; p++)
                    {
                        sum += ar[p] * panel[((long)p * tail) + j];
                    }

                    c[((long)row * n) + col + j] = sum;
                }
            }
        }
    }

    private sealed unsafe class VectorJob
    {
        public float* A;
        public float* B;
        public float* C;
        public int M;
        public int K;
        public int N;
        public int Threads;

        public void Invoke(int id)
        {
            int start = (int)((long)id * M / Threads);
            int end = (int)((long)(id + 1) * M / Threads);
            MultiplyVectorRange(A, B, C, start, end, K, N);
        }
    }

    private static unsafe void MultiplyVectorRange(float* a, float* b, float* c, int row0, int row1, int k, int n)
    {
        int width = Vector<float>.Count;
        for (int i = row0; i < row1; i++)
        {
            float* cRow = c + ((long)i * n);
            float* aRow = a + ((long)i * k);
            for (int p = 0; p < k; p++)
            {
                float av = aRow[p];
                var aVec = new Vector<float>(av);
                float* bRow = b + ((long)p * n);
                int j = 0;
                for (; j + width <= n; j += width)
                {
                    var cv = new Vector<float>(new ReadOnlySpan<float>(cRow + j, width));
                    var bv = new Vector<float>(new ReadOnlySpan<float>(bRow + j, width));
                    (cv + (aVec * bv)).CopyTo(new Span<float>(cRow + j, width));
                }

                for (; j < n; j++)
                {
                    cRow[j] += av * bRow[j];
                }
            }
        }
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
