using System.Buffers.Binary;

namespace Magika;

/// <summary>
/// ZIP central-directory subtype check for MSIX, matching upstream google/magika
/// <c>resolvers::magic::zip</c>: a valid archive whose directory names include both
/// <c>AppxManifest.xml</c> and <c>AppxBlockMap.xml</c> (exact entry names).
/// </summary>
internal static class ZipMsixResolver
{
    /// <summary>Trailing window size used by upstream <c>ZIP_TAIL_BYTES</c>.</summary>
    internal const int TailBytes = 16 * 1024;

    /// <summary>Largest central directory re-read when the initial tail does not hold it.</summary>
    internal const int DirectoryMaxBytes = 256 * 1024;

    private const int CentralHeaderLen = 46;
    private const int EocdLen = 22;
    private const int Zip64LocatorLen = 20;
    private const int Zip64EocdLen = 56;

    private const byte FlagZip64 = 1 << 0;
    private const byte FlagMultidisk = 1 << 1;
    private const byte FlagCdNotHeld = 1 << 2;
    private const byte FlagMalformed = 1 << 3;

    private static ReadOnlySpan<byte> LocalSignature => "PK\u0003\u0004"u8;
    private static ReadOnlySpan<byte> CentralSignature => "PK\u0001\u0002"u8;
    private static ReadOnlySpan<byte> EocdSignature => "PK\u0005\u0006"u8;
    private static ReadOnlySpan<byte> Zip64LocatorSignature => "PK\u0006\u0007"u8;
    private static ReadOnlySpan<byte> Zip64EocdSignature => "PK\u0006\u0006"u8;
    private static ReadOnlySpan<byte> AppxManifest => "AppxManifest.xml"u8;
    private static ReadOnlySpan<byte> AppxBlockMap => "AppxBlockMap.xml"u8;

    /// <summary>Whether the prefix opens like a ZIP that needs a tail for directory facts.</summary>
    internal static bool WantsTail(ReadOnlySpan<byte> prefix) =>
        prefix.StartsWith(LocalSignature) || prefix.StartsWith(EocdSignature);

    /// <summary>
    /// Returns true when <paramref name="prefix"/> / <paramref name="tail"/> describe a valid ZIP
    /// whose central directory lists both AppxManifest.xml and AppxBlockMap.xml.
    /// </summary>
    internal static bool IsMsix(ReadOnlySpan<byte> prefix, long size, ReadOnlySpan<byte> tail)
    {
        if (size < 0 || !WantsTail(prefix))
        {
            return false;
        }

        if (!TryAnalyze(prefix, (ulong)size, tail, out bool valid, out bool hasManifest, out bool hasBlockMap))
        {
            return false;
        }

        return valid && hasManifest && hasBlockMap;
    }

    /// <summary>
    /// Absolute offset at which to start reading so the central directory is held, when the
    /// current <paramref name="tail"/> (ending at <paramref name="size"/>) holds the EOCD but
    /// not the directory. Mirrors upstream <c>facts::directory_start</c>.
    /// </summary>
    internal static long? DirectoryStart(ReadOnlySpan<byte> tail, long size)
    {
        if (size < 0 || !TryFindEocd(tail, out int eocd))
        {
            return null;
        }

        Eocd record = Eocd.Decode(tail, eocd);
        if (!TryTailStart((ulong)size, tail.Length, out ulong tailStart))
        {
            return null;
        }

        ulong eocdAbs = tailStart + (ulong)eocd;
        ulong cdSize;
        ulong cdEndAbs;
        if (TryZip64Directory(tail, eocd, tailStart, out ulong entries64, out ulong cdSize64, out _, out ulong zip64End))
        {
            _ = entries64;
            cdSize = cdSize64;
            cdEndAbs = zip64End;
        }
        else if (record.Flags != 0)
        {
            return null;
        }
        else
        {
            cdSize = record.CdSize;
            cdEndAbs = eocdAbs;
        }

        if (cdSize > DirectoryMaxBytes)
        {
            return null;
        }

        if (cdEndAbs < cdSize)
        {
            return null;
        }

        ulong start = cdEndAbs - cdSize;
        return start < tailStart ? (long)start : null;
    }

    private static bool TryAnalyze(
        ReadOnlySpan<byte> first,
        ulong size,
        ReadOnlySpan<byte> tail,
        out bool valid,
        out bool hasManifest,
        out bool hasBlockMap)
    {
        valid = false;
        hasManifest = false;
        hasBlockMap = false;

        if (!TryFindEocd(tail, out int eocd))
        {
            return false;
        }

        if (!TryTailStart(size, tail.Length, out ulong tailStart))
        {
            return false;
        }

        Eocd record = Eocd.Decode(tail, eocd);
        byte flags = record.Flags;
        if ((flags & FlagMultidisk) != 0)
        {
            return true;
        }

        ulong eocdAbs = tailStart + (ulong)eocd;
        ulong entries;
        ulong cdSize;
        ulong cdEndAbs;
        if (TryZip64Directory(tail, eocd, tailStart, out ulong entries64, out ulong cdSize64, out _, out ulong zip64End))
        {
            flags |= FlagZip64;
            entries = entries64;
            cdSize = cdSize64;
            cdEndAbs = zip64End;
        }
        else if ((record.Flags & FlagZip64) != 0)
        {
            return true;
        }
        else
        {
            entries = record.Entries;
            cdSize = record.CdSize;
            cdEndAbs = eocdAbs;
        }

        if (cdEndAbs < cdSize)
        {
            flags |= FlagMalformed;
            valid = false;
            return true;
        }

        ulong cdAbs = cdEndAbs - cdSize;
        ReadOnlySpan<byte> directory;
        if (cdAbs >= tailStart)
        {
            int start = (int)(cdAbs - tailStart);
            int end = (int)(cdEndAbs - tailStart);
            if ((uint)end > (uint)tail.Length || start > end)
            {
                flags |= FlagMalformed;
                return true;
            }

            directory = tail[start..end];
        }
        else if (cdEndAbs <= (ulong)first.Length)
        {
            directory = first[(int)cdAbs..(int)cdEndAbs];
        }
        else
        {
            flags |= FlagCdNotHeld;
            return true;
        }

        WalkNames(directory, entries, ref hasManifest, ref hasBlockMap, ref flags);
        valid = (flags & (FlagMultidisk | FlagCdNotHeld | FlagMalformed)) == 0;
        return true;
    }

    private static void WalkNames(
        ReadOnlySpan<byte> directory,
        ulong entries,
        ref bool hasManifest,
        ref bool hasBlockMap,
        ref byte flags)
    {
        // Bound the walk by the held directory even if entries is hostile.
        ulong limit = entries;
        if (limit > (ulong)(directory.Length / CentralHeaderLen + 1))
        {
            limit = (ulong)(directory.Length / CentralHeaderLen + 1);
        }

        int at = 0;
        for (ulong i = 0; i < limit; i++)
        {
            if (!TryEntryName(directory, at, out ReadOnlySpan<byte> name, out int entryLen))
            {
                flags |= FlagMalformed;
                return;
            }

            at += entryLen;
            if (name.SequenceEqual(AppxManifest))
            {
                hasManifest = true;
            }
            else if (name.SequenceEqual(AppxBlockMap))
            {
                hasBlockMap = true;
            }

            if (hasManifest && hasBlockMap)
            {
                return;
            }
        }
    }

    private static bool TryEntryName(ReadOnlySpan<byte> directory, int at, out ReadOnlySpan<byte> name, out int entryLen)
    {
        name = default;
        entryLen = 0;
        if ((uint)at > (uint)directory.Length || directory.Length - at < CentralHeaderLen)
        {
            return false;
        }

        ReadOnlySpan<byte> header = directory.Slice(at, CentralHeaderLen);
        if (!header.StartsWith(CentralSignature))
        {
            return false;
        }

        int nameLen = BinaryPrimitives.ReadUInt16LittleEndian(header[28..]);
        int trailer = BinaryPrimitives.ReadUInt16LittleEndian(header[30..])
            + BinaryPrimitives.ReadUInt16LittleEndian(header[32..]);
        int nameStart = at + CentralHeaderLen;
        if (nameStart > directory.Length || nameLen > directory.Length - nameStart)
        {
            return false;
        }

        int nameEnd = nameStart + nameLen;
        if (trailer > directory.Length - nameEnd)
        {
            return false;
        }

        name = directory.Slice(nameStart, nameLen);
        entryLen = CentralHeaderLen + nameLen + trailer;
        return true;
    }

    private static bool TryFindEocd(ReadOnlySpan<byte> tail, out int eocd)
    {
        eocd = -1;
        if (tail.Length < EocdLen)
        {
            return false;
        }

        int last = tail.Length - EocdLen;
        for (int at = last; at >= 0; at--)
        {
            if (!tail[at..].StartsWith(EocdSignature))
            {
                continue;
            }

            ushort commentLen = BinaryPrimitives.ReadUInt16LittleEndian(tail[(at + 20)..]);
            if (commentLen == tail.Length - at - EocdLen)
            {
                eocd = at;
                return true;
            }
        }

        return false;
    }

    private static bool TryZip64Directory(
        ReadOnlySpan<byte> tail,
        int eocd,
        ulong tailStart,
        out ulong entries,
        out ulong cdSize,
        out ulong cdOffset,
        out ulong cdEndAbs)
    {
        entries = 0;
        cdSize = 0;
        cdOffset = 0;
        cdEndAbs = 0;
        if (eocd < Zip64LocatorLen)
        {
            return false;
        }

        int locator = eocd - Zip64LocatorLen;
        if (!tail[locator..].StartsWith(Zip64LocatorSignature))
        {
            return false;
        }

        ulong recordAbs = BinaryPrimitives.ReadUInt64LittleEndian(tail[(locator + 8)..]);
        if (recordAbs < tailStart)
        {
            return false;
        }

        ulong at64 = recordAbs - tailStart;
        if (at64 > (ulong)int.MaxValue)
        {
            return false;
        }

        int at = (int)at64;
        if (at > tail.Length || tail.Length - at < Zip64EocdLen)
        {
            return false;
        }

        ReadOnlySpan<byte> record = tail.Slice(at, Zip64EocdLen);
        if (!record.StartsWith(Zip64EocdSignature))
        {
            return false;
        }

        entries = BinaryPrimitives.ReadUInt64LittleEndian(record[32..]);
        cdSize = BinaryPrimitives.ReadUInt64LittleEndian(record[40..]);
        cdOffset = BinaryPrimitives.ReadUInt64LittleEndian(record[48..]);
        cdEndAbs = recordAbs;
        return true;
    }

    private static bool TryTailStart(ulong size, int tailLength, out ulong tailStart)
    {
        tailStart = 0;
        if (tailLength < 0 || size < (ulong)tailLength)
        {
            return false;
        }

        tailStart = size - (ulong)tailLength;
        return true;
    }

    private readonly struct Eocd
    {
        public ushort Entries { get; init; }
        public uint CdSize { get; init; }
        public uint CdOffset { get; init; }
        public byte Flags { get; init; }

        public static Eocd Decode(ReadOnlySpan<byte> tail, int eocd)
        {
            ReadOnlySpan<byte> record = tail.Slice(eocd, EocdLen);
            ushort disk = BinaryPrimitives.ReadUInt16LittleEndian(record[4..]);
            ushort cdDisk = BinaryPrimitives.ReadUInt16LittleEndian(record[6..]);
            ushort entriesDisk = BinaryPrimitives.ReadUInt16LittleEndian(record[8..]);
            ushort entries = BinaryPrimitives.ReadUInt16LittleEndian(record[10..]);
            uint cdSize = BinaryPrimitives.ReadUInt32LittleEndian(record[12..]);
            uint cdOffset = BinaryPrimitives.ReadUInt32LittleEndian(record[16..]);
            bool locator = eocd >= Zip64LocatorLen
                && tail[(eocd - Zip64LocatorLen)..].StartsWith(Zip64LocatorSignature);

            byte flags = 0;
            if (disk == ushort.MaxValue
                || cdDisk == ushort.MaxValue
                || entriesDisk == ushort.MaxValue
                || entries == ushort.MaxValue
                || cdSize == uint.MaxValue
                || cdOffset == uint.MaxValue
                || locator)
            {
                flags |= FlagZip64;
            }

            if (disk != 0 || cdDisk != 0 || entriesDisk != entries)
            {
                flags |= FlagMultidisk;
            }

            return new Eocd
            {
                Entries = entries,
                CdSize = cdSize,
                CdOffset = cdOffset,
                Flags = flags,
            };
        }
    }
}
