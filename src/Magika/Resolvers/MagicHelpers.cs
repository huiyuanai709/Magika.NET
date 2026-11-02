using System.Buffers.Binary;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Magika.Resolvers;

/// <summary>Shared span helpers for the google/magika magic cascade (zero-allocation hot path).</summary>
internal static class MagicHelpers
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool Starts(ReadOnlySpan<byte> hay, ReadOnlySpan<byte> sig) => hay.StartsWith(sig);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool EqAt(ReadOnlySpan<byte> hay, int off, ReadOnlySpan<byte> sig)
    {
        if ((uint)off > (uint)hay.Length || hay.Length - off < sig.Length)
        {
            return false;
        }

        return hay.Slice(off, sig.Length).SequenceEqual(sig);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static byte? U8At(ReadOnlySpan<byte> hay, int off) =>
        (uint)off < (uint)hay.Length ? hay[off] : null;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ushort? Le16(ReadOnlySpan<byte> hay, int off)
    {
        if ((uint)off > (uint)hay.Length || hay.Length - off < 2)
        {
            return null;
        }

        return BinaryPrimitives.ReadUInt16LittleEndian(hay.Slice(off));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ushort? Be16(ReadOnlySpan<byte> hay, int off)
    {
        if ((uint)off > (uint)hay.Length || hay.Length - off < 2)
        {
            return null;
        }

        return BinaryPrimitives.ReadUInt16BigEndian(hay.Slice(off));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static uint? Le32(ReadOnlySpan<byte> hay, int off)
    {
        if ((uint)off > (uint)hay.Length || hay.Length - off < 4)
        {
            return null;
        }

        return BinaryPrimitives.ReadUInt32LittleEndian(hay.Slice(off));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static uint? Be32(ReadOnlySpan<byte> hay, int off)
    {
        if ((uint)off > (uint)hay.Length || hay.Length - off < 4)
        {
            return null;
        }

        return BinaryPrimitives.ReadUInt32BigEndian(hay.Slice(off));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static uint TagU32(ReadOnlySpan<byte> tag4) =>
        BinaryPrimitives.ReadUInt32BigEndian(tag4);

    /// <summary>Whether <paramref name="needle"/> occurs starting at some offset in <c>lo..=hi</c>.</summary>
    internal static bool ScanRange(ReadOnlySpan<byte> hay, int lo, int hi, ReadOnlySpan<byte> needle)
    {
        if (needle.IsEmpty || lo < 0)
        {
            return false;
        }

        int end = Math.Min(hay.Length, hi + needle.Length);
        if (lo >= end)
        {
            return false;
        }

        return hay.Slice(lo, end - lo).IndexOf(needle) >= 0;
    }

    /// <summary>Mixed-endian GUID bytes matching upstream <c>guid!</c> (Data1/2/3 LE, Data4 BE).</summary>
    internal static byte[] GuidBytes(string guid)
    {
        // "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
        Span<byte> raw = stackalloc byte[16];
        ReadOnlySpan<char> s = guid.AsSpan();
        static int Nibble(char c) => char.IsAsciiHexDigit(c)
            ? int.Parse(c.ToString(), NumberStyles.HexNumber)
            : throw new ArgumentException("invalid GUID digit");

        static byte Byte(ReadOnlySpan<char> s, int idx) =>
            (byte)((Nibble(s[idx]) << 4) | Nibble(s[idx + 1]));

        raw[0] = Byte(s, 6);
        raw[1] = Byte(s, 4);
        raw[2] = Byte(s, 2);
        raw[3] = Byte(s, 0);
        raw[4] = Byte(s, 11);
        raw[5] = Byte(s, 9);
        raw[6] = Byte(s, 16);
        raw[7] = Byte(s, 14);
        raw[8] = Byte(s, 19);
        raw[9] = Byte(s, 21);
        raw[10] = Byte(s, 24);
        raw[11] = Byte(s, 26);
        raw[12] = Byte(s, 28);
        raw[13] = Byte(s, 30);
        raw[14] = Byte(s, 32);
        raw[15] = Byte(s, 34);
        return raw.ToArray();
    }
}

/// <summary>Per-input scan context for the magic cascade.</summary>
internal readonly ref struct MagicCtx
{
    public ReadOnlySpan<byte> Prefix { get; init; }
    public ulong Size { get; init; }
    public ReadOnlySpan<byte> Tail { get; init; }
    public bool HasTail { get; init; }
}
