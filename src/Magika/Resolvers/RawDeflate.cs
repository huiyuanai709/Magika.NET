// Port of Micha Mettke's sinfl (https://github.com/fxfactorial/sdefl) — MIT / public domain.
// Inflates raw DEFLATE (RFC 1951) without System.IO.Compression so Native AOT
// avx2 SKUs do not link zlib-ng's AVX-512 kernels.

using System.IO;
using System.Runtime.CompilerServices;

namespace Magika.Resolvers;

internal static class RawDeflate
{
    /// <summary>
    /// Inflates raw DEFLATE into <paramref name="destination"/>, stopping when the
    /// buffer is full or the stream ends. Returns <c>false</c> on malformed input.
    /// </summary>
    internal static bool TryInflate(ReadOnlySpan<byte> source, Span<byte> destination, out int written)
    {
        written = 0;
        if (destination.IsEmpty)
        {
            return true;
        }

        try
        {
            written = Inflate(source, destination);
            return true;
        }
        catch (InvalidDataException)
        {
            written = 0;
            return false;
        }
    }

    private static int Inflate(ReadOnlySpan<byte> source, Span<byte> destination)
    {
        ReadOnlySpan<byte> order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
        ReadOnlySpan<ushort> dbase =
        [
            1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
            257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0, 0,
        ];
        ReadOnlySpan<byte> dbits =
        [
            0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
            10, 10, 11, 11, 12, 12, 13, 13, 0, 0,
        ];
        ReadOnlySpan<ushort> lbase =
        [
            3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35,
            43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0,
        ];
        ReadOnlySpan<byte> lbits =
        [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4,
            4, 4, 4, 5, 5, 5, 5, 0, 0, 0,
        ];

        var s = new State();
        int inAt = 0;
        int outAt = 0;
        int last = 0;
        StateKind state = StateKind.Hdr;

        Get(ref s, source, ref inAt, 0);
        while (inAt < source.Length || s.BitCount != 0)
        {
            switch (state)
            {
                case StateKind.Hdr:
                {
                    last = Get(ref s, source, ref inAt, 1);
                    int type = Get(ref s, source, ref inAt, 2);
                    state = type switch
                    {
                        0 => StateKind.Stored,
                        1 => StateKind.Fixed,
                        2 => StateKind.Dyn,
                        _ => throw new InvalidDataException(),
                    };
                    break;
                }
                case StateKind.Stored:
                {
                    Get(ref s, source, ref inAt, s.BitCount & 7);
                    int len = Get(ref s, source, ref inAt, 16);
                    _ = Get(ref s, source, ref inAt, 16);
                    inAt -= 2;
                    s.BitCount = 0;
                    if (len <= 0 || len > source.Length - inAt)
                    {
                        return outAt;
                    }

                    int copy = Math.Min(len, destination.Length - outAt);
                    source.Slice(inAt, copy).CopyTo(destination.Slice(outAt));
                    inAt += len;
                    outAt += copy;
                    if (outAt == destination.Length)
                    {
                        return outAt;
                    }

                    state = StateKind.Hdr;
                    break;
                }
                case StateKind.Fixed:
                {
                    Span<byte> lens = stackalloc byte[288 + 32];
                    lens[..144].Fill(8);
                    lens.Slice(144, 112).Fill(9);
                    lens.Slice(256, 24).Fill(7);
                    lens.Slice(280, 8).Fill(8);
                    lens.Slice(288, 32).Fill(5);
                    s.TLit = Build(s.Lits, lens[..288]);
                    s.TDist = Build(s.Dsts, lens.Slice(288, 32));
                    state = StateKind.Blk;
                    break;
                }
                case StateKind.Dyn:
                {
                    int nlit = 257 + Get(ref s, source, ref inAt, 5);
                    int ndist = 1 + Get(ref s, source, ref inAt, 5);
                    int nlen = 4 + Get(ref s, source, ref inAt, 4);
                    Span<byte> nlens = stackalloc byte[19];
                    nlens.Clear();
                    for (int n = 0; n < nlen; n++)
                    {
                        nlens[order[n]] = (byte)Get(ref s, source, ref inAt, 3);
                    }

                    s.TLen = Build(s.Lens, nlens);

                    Span<byte> lens = stackalloc byte[288 + 32];
                    lens.Clear();
                    for (int n = 0; n < nlit + ndist;)
                    {
                        int sym = Decode(ref s, source, ref inAt, s.Lens, s.TLen);
                        switch (sym)
                        {
                            case 16:
                            {
                                if (n == 0)
                                {
                                    throw new InvalidDataException();
                                }

                                byte prev = lens[n - 1];
                                for (int i = 3 + Get(ref s, source, ref inAt, 2); i > 0; i--, n++)
                                {
                                    if (n >= lens.Length)
                                    {
                                        throw new InvalidDataException();
                                    }

                                    lens[n] = prev;
                                }

                                break;
                            }
                            case 17:
                                for (int i = 3 + Get(ref s, source, ref inAt, 3); i > 0; i--, n++)
                                {
                                    if (n >= lens.Length)
                                    {
                                        throw new InvalidDataException();
                                    }

                                    lens[n] = 0;
                                }

                                break;
                            case 18:
                                for (int i = 11 + Get(ref s, source, ref inAt, 7); i > 0; i--, n++)
                                {
                                    if (n >= lens.Length)
                                    {
                                        throw new InvalidDataException();
                                    }

                                    lens[n] = 0;
                                }

                                break;
                            default:
                                if (n >= lens.Length || (uint)sym > 15)
                                {
                                    throw new InvalidDataException();
                                }

                                lens[n++] = (byte)sym;
                                break;
                        }
                    }

                    s.TLit = Build(s.Lits, lens[..nlit]);
                    s.TDist = Build(s.Dsts, lens.Slice(nlit, ndist));
                    state = StateKind.Blk;
                    break;
                }
                case StateKind.Blk:
                {
                    int sym = Decode(ref s, source, ref inAt, s.Lits, s.TLit);
                    if (sym > 256)
                    {
                        sym -= 257;
                        if ((uint)sym >= (uint)lbits.Length)
                        {
                            throw new InvalidDataException();
                        }

                        int len = Get(ref s, source, ref inAt, lbits[sym]) + lbase[sym];
                        int dsym = Decode(ref s, source, ref inAt, s.Dsts, s.TDist);
                        if ((uint)dsym >= (uint)dbits.Length)
                        {
                            throw new InvalidDataException();
                        }

                        int offs = Get(ref s, source, ref inAt, dbits[dsym]) + dbase[dsym];
                        if (offs <= 0 || offs > outAt)
                        {
                            return outAt;
                        }

                        while (len-- > 0)
                        {
                            if (outAt == destination.Length)
                            {
                                return outAt;
                            }

                            destination[outAt] = destination[outAt - offs];
                            outAt++;
                        }
                    }
                    else if (sym == 256)
                    {
                        if (last != 0)
                        {
                            return outAt;
                        }

                        state = StateKind.Hdr;
                    }
                    else
                    {
                        if (outAt == destination.Length)
                        {
                            return outAt;
                        }

                        destination[outAt++] = (byte)sym;
                    }

                    break;
                }
            }
        }

        return outAt;
    }

    private enum StateKind
    {
        Hdr,
        Stored,
        Fixed,
        Dyn,
        Blk,
    }

    private struct State
    {
        public int Bits;
        public int BitCount;
        public int TLit;
        public int TDist;
        public int TLen;
        public readonly uint[] Lits;
        public readonly uint[] Dsts;
        public readonly uint[] Lens;

        public State()
        {
            Bits = 0;
            BitCount = 0;
            TLit = 0;
            TDist = 0;
            TLen = 0;
            Lits = new uint[288];
            Dsts = new uint[32];
            Lens = new uint[19];
        }
    }

    private static ReadOnlySpan<byte> Mirror =>
    [
        0, 128, 64, 192, 32, 160, 96, 224, 16, 144, 80, 208, 48, 176, 112, 240,
        8, 136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
        4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244,
        12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
        2, 130, 66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242,
        10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
        6, 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246,
        14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
        1, 129, 65, 193, 33, 161, 97, 225, 17, 145, 81, 209, 49, 177, 113, 241,
        9, 137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
        5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245,
        13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
        3, 131, 67, 195, 35, 163, 99, 227, 19, 147, 83, 211, 51, 179, 115, 243,
        11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
        7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247,
        15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255,
    ];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Get(ref State s, ReadOnlySpan<byte> source, ref int inAt, int n)
    {
        int v = s.Bits & ((1 << n) - 1);
        s.Bits >>= n;
        s.BitCount -= n;
        if (s.BitCount < 0)
        {
            s.BitCount = 0;
        }

        while (s.BitCount < 16 && inAt < source.Length)
        {
            s.Bits |= source[inAt++] << s.BitCount;
            s.BitCount += 8;
        }

        return v;
    }

    private static int Build(Span<uint> tree, ReadOnlySpan<byte> lens)
    {
        Span<int> cnt = stackalloc int[16];
        Span<int> first = stackalloc int[16];
        Span<int> codes = stackalloc int[16];
        cnt.Clear();
        first.Clear();
        codes.Clear();
        for (int n = 0; n < lens.Length; n++)
        {
            cnt[lens[n]]++;
        }

        for (int n = 1; n <= 15; n++)
        {
            codes[n] = (codes[n - 1] + cnt[n - 1]) << 1;
            first[n] = first[n - 1] + cnt[n - 1];
        }

        for (int n = 0; n < lens.Length; n++)
        {
            int len = lens[n];
            if (len == 0)
            {
                continue;
            }

            int code = codes[len]++;
            int slot = first[len]++;
            tree[slot] = (uint)((code << (32 - len)) | (n << 4) | len);
        }

        return first[15];
    }

    private static int Decode(ref State s, ReadOnlySpan<byte> source, ref int inAt, Span<uint> tree, int max)
    {
        uint lo = 0;
        uint hi = (uint)max;
        uint search = ((uint)Rev16(s.Bits) << 16) | 0xffff;
        while (lo < hi)
        {
            uint guess = (lo + hi) / 2;
            if (search < tree[(int)guess])
            {
                hi = guess;
            }
            else
            {
                lo = guess + 1;
            }
        }

        if (lo == 0)
        {
            throw new InvalidDataException();
        }

        uint key = tree[(int)lo - 1];
        Get(ref s, source, ref inAt, (int)(key & 0x0f));
        return (int)((key >> 4) & 0x0fff);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Rev16(int n) => (Mirror[n & 0xff] << 8) | Mirror[(n >> 8) & 0xff];
}
