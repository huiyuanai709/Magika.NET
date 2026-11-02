using System.Buffers;
using System.Buffers.Binary;
using System.IO.Compression;
using System.Runtime.CompilerServices;

namespace Magika.Resolvers;

/// <summary>
/// ZIP facts matching upstream <c>resolvers::facts::zip</c>: one EOCD/central-directory walk
/// produces validity, a <c>\n</c>-framed names view (plus optional stored mimetype line), and a
/// first-entry view (name + inflated payload). Subtype checks share this single parse.
/// </summary>
internal static class ZipFacts
{
    internal const int TailBytes = 16 * 1024;
    internal const int DirectoryMaxBytes = 256 * 1024;
    internal const int NamesBytes = 4096;
    internal const int FirstEntryBytes = 16 * 1024;
    internal const int PrefixLimit = 4096;

    private const int CentralHeaderLen = 46;
    private const int LocalHeaderLen = 30;
    private const int EocdLen = 22;
    private const int Zip64LocatorLen = 20;
    private const int Zip64EocdLen = 56;
    private const int MimetypeMaxLen = 128;

    private const byte FlagZip64 = 1 << 0;
    private const byte FlagMultidisk = 1 << 1;
    private const byte FlagCdNotHeld = 1 << 2;
    private const byte FlagMalformed = 1 << 3;
    private const byte FlagFirstEntryFilled = 1 << 4;
    private const byte FlagFirstEntryUndecodable = 1 << 5;

    private static ReadOnlySpan<byte> LocalSignature => "PK\u0003\u0004"u8;
    private static ReadOnlySpan<byte> CentralSignature => "PK\u0001\u0002"u8;
    private static ReadOnlySpan<byte> EocdSignature => "PK\u0005\u0006"u8;
    private static ReadOnlySpan<byte> Zip64LocatorSignature => "PK\u0006\u0007"u8;
    private static ReadOnlySpan<byte> Zip64EocdSignature => "PK\u0006\u0006"u8;
    private static ReadOnlySpan<byte> MimetypeName => "mimetype"u8;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool WantsTail(ReadOnlySpan<byte> prefix) =>
        prefix.StartsWith(LocalSignature) || prefix.StartsWith(EocdSignature);

    /// <summary>Absolute offset to re-read so the central directory is held.</summary>
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
        if (TryZip64Directory(tail, eocd, tailStart, out _, out ulong cdSize64, out _, out ulong zip64End))
        {
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

        if (cdSize > DirectoryMaxBytes || cdEndAbs < cdSize)
        {
            return null;
        }

        ulong start = cdEndAbs - cdSize;
        return start < tailStart ? (long)start : null;
    }

    /// <summary>Analyzes ZIP facts into rented buffers. Caller must return the leases.</summary>
    internal static Analysis Analyze(ReadOnlySpan<byte> prefix, ulong size, ReadOnlySpan<byte> tail)
    {
        byte[] namesBuf = ArrayPool<byte>.Shared.Rent(NamesBytes);
        byte[] firstBuf = ArrayPool<byte>.Shared.Rent(FirstEntryBytes);
        namesBuf.AsSpan(0, NamesBytes).Clear();
        firstBuf.AsSpan(0, FirstEntryBytes).Clear();

        var names = new ViewWriter(namesBuf.AsSpan(0, NamesBytes));
        bool valid = false;
        bool hasDirNames = false;
        ushort entries = 0;
        byte flags = 0;

        if (TryStoredMimetype(prefix, out ReadOnlySpan<byte> mime))
        {
            if (names.Fits("mimetype="u8, mime))
            {
                names.Line("mimetype="u8, mime, countEntry: false);
            }
        }

        if (TryFindEocd(tail, out int eocd)
            && TryWalkDirectory(prefix, size, tail, eocd, ref names, out entries, out flags, out valid))
        {
            hasDirNames = valid && entries >= 1 && names.EntryCount >= 1;
        }

        int namesLen = names.Finish();
        int firstLen = WriteFirstEntry(prefix, firstBuf.AsSpan(0, FirstEntryBytes), ref flags);

        return new Analysis(namesBuf, namesLen, firstBuf, firstLen, valid, hasDirNames, flags);
    }

    internal readonly struct Analysis : IDisposable
    {
        private readonly byte[] _namesArray;
        private readonly byte[] _firstArray;

        public Analysis(byte[] namesArray, int namesLen, byte[] firstArray, int firstLen, bool valid, bool hasDirNames, byte flags)
        {
            _namesArray = namesArray;
            _firstArray = firstArray;
            NamesLen = namesLen;
            FirstLen = firstLen;
            Valid = valid;
            HasDirNames = hasDirNames;
            Flags = flags;
        }

        public int NamesLen { get; }
        public int FirstLen { get; }
        public bool Valid { get; }
        public bool HasDirNames { get; }
        public byte Flags { get; }

        public ReadOnlySpan<byte> Names => _namesArray.AsSpan(0, NamesLen);
        public ReadOnlySpan<byte> FirstEntry => _firstArray.AsSpan(0, FirstLen);

        public bool HasLine(ReadOnlySpan<byte> lineWithNewlines) => Names.IndexOf(lineWithNewlines) >= 0;

        public bool NamesStartsWith(ReadOnlySpan<byte> prefix) => Names.StartsWith(prefix);

        public void Dispose()
        {
            ArrayPool<byte>.Shared.Return(_namesArray);
            ArrayPool<byte>.Shared.Return(_firstArray);
        }
    }

    private ref struct ViewWriter
    {
        private readonly Span<byte> _out;
        private int _len;
        public int EntryCount { get; private set; }

        public ViewWriter(Span<byte> output)
        {
            _out = output;
            _len = 0;
            EntryCount = 0;
        }

        public bool Fits(ReadOnlySpan<byte> prefix, ReadOnlySpan<byte> text) =>
            _len + 1 + prefix.Length + text.Length < _out.Length;

        public void Line(ReadOnlySpan<byte> prefix, ReadOnlySpan<byte> text, bool countEntry = true)
        {
            _out[_len++] = (byte)'\n';
            prefix.CopyTo(_out.Slice(_len));
            _len += prefix.Length;
            for (int i = 0; i < text.Length; i++)
            {
                byte b = text[i];
                _out[_len++] = b == 0 || b == (byte)'\n' ? (byte)1 : b;
            }

            if (countEntry)
            {
                EntryCount++;
            }
        }

        public int Finish()
        {
            if (_len == 0)
            {
                return 0;
            }

            _out[_len] = (byte)'\n';
            return _len + 1;
        }
    }

    private static bool TryStoredMimetype(ReadOnlySpan<byte> first, out ReadOnlySpan<byte> payload)
    {
        payload = default;
        if (!first.StartsWith(LocalSignature))
        {
            return false;
        }

        if (Le16(first, 8) != 0 || Le16(first, 26) != 8)
        {
            return false;
        }

        uint length = Le32(first, 22);
        if (length < 1 || length > MimetypeMaxLen)
        {
            return false;
        }

        int extraLen = Le16(first, 28);
        int nameEnd = LocalHeaderLen + 8;
        if (first.Length < nameEnd || !first.Slice(LocalHeaderLen, 8).SequenceEqual(MimetypeName))
        {
            return false;
        }

        int dataStart = nameEnd + extraLen;
        int dataEnd = dataStart + (int)length;
        if ((uint)dataEnd > (uint)first.Length)
        {
            return false;
        }

        payload = first.Slice(dataStart, (int)length);
        return true;
    }

    private static bool TryWalkDirectory(
        ReadOnlySpan<byte> first,
        ulong size,
        ReadOnlySpan<byte> tail,
        int eocd,
        ref ViewWriter names,
        out ushort entriesOut,
        out byte flags,
        out bool valid)
    {
        entriesOut = 0;
        flags = 0;
        valid = false;

        if (!TryTailStart(size, tail.Length, out ulong tailStart))
        {
            return false;
        }

        Eocd record = Eocd.Decode(tail, eocd);
        flags = record.Flags;
        entriesOut = record.Entries;
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

        entriesOut = (ushort)Math.Min(entries, ushort.MaxValue);

        if (cdEndAbs < cdSize)
        {
            flags |= FlagMalformed;
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

            directory = tail.Slice(start, end - start);
        }
        else if (cdEndAbs <= (ulong)first.Length)
        {
            directory = first.Slice((int)cdAbs, (int)cdSize);
        }
        else
        {
            flags |= FlagCdNotHeld;
            return true;
        }

        WalkNames(directory, entries, ref names, ref flags);
        valid = (flags & (FlagMultidisk | FlagCdNotHeld | FlagMalformed)) == 0;
        return true;
    }

    private static void WalkNames(ReadOnlySpan<byte> directory, ulong entries, ref ViewWriter names, ref byte flags)
    {
        ulong limit = entries;
        ulong maxBySize = (ulong)(directory.Length / CentralHeaderLen + 1);
        if (limit > maxBySize)
        {
            limit = maxBySize;
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
            if (names.Fits(ReadOnlySpan<byte>.Empty, name))
            {
                names.Line(ReadOnlySpan<byte>.Empty, name);
            }
            else
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

        int nameLen = BinaryPrimitives.ReadUInt16LittleEndian(header.Slice(28));
        int trailer = BinaryPrimitives.ReadUInt16LittleEndian(header.Slice(30))
            + BinaryPrimitives.ReadUInt16LittleEndian(header.Slice(32));
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

    private static int WriteFirstEntry(ReadOnlySpan<byte> first, Span<byte> output, ref byte flags)
    {
        if (!TryFirstEntryParts(first, out ReadOnlySpan<byte> name, out ReadOnlySpan<byte> data, out bool deflated))
        {
            return 0;
        }

        int head = name.Length + 1;
        if (head >= output.Length)
        {
            return 0;
        }

        for (int i = 0; i < name.Length; i++)
        {
            byte b = name[i];
            output[i] = b == 0 || b == (byte)'\n' ? (byte)1 : b;
        }

        output[name.Length] = (byte)'\n';
        Span<byte> space = output.Slice(head);
        int written;
        bool complete;
        if (deflated)
        {
            // Raw DEFLATE (RFC 1951), matching miniz TINFL non-wrapping buffer.
            byte[]? rented = null;
            try
            {
                rented = ArrayPool<byte>.Shared.Rent(data.Length);
                data.CopyTo(rented);
                using var ms = new MemoryStream(rented, 0, data.Length, writable: false, publiclyVisible: true);
                using var ds = new DeflateStream(ms, CompressionMode.Decompress, leaveOpen: true);
                written = ds.Read(space);
                if (written == space.Length)
                {
                    flags |= FlagFirstEntryFilled;
                    return FirstEntryBytes;
                }

                complete = true;
            }
            catch
            {
                flags |= FlagFirstEntryUndecodable;
                return head;
            }
            finally
            {
                if (rented is not null)
                {
                    ArrayPool<byte>.Shared.Return(rented);
                }
            }
        }
        else
        {
            written = Math.Min(data.Length, space.Length);
            data.Slice(0, written).CopyTo(space);
            complete = true;
            if (written == space.Length)
            {
                flags |= FlagFirstEntryFilled;
                return FirstEntryBytes;
            }
        }

        if (!complete)
        {
            space.Slice(0, written).Clear();
            flags |= FlagFirstEntryUndecodable;
            return head;
        }

        return head + written;
    }

    private static bool TryFirstEntryParts(
        ReadOnlySpan<byte> first,
        out ReadOnlySpan<byte> name,
        out ReadOnlySpan<byte> data,
        out bool deflated)
    {
        name = default;
        data = default;
        deflated = false;
        if (!first.StartsWith(LocalSignature))
        {
            return false;
        }

        ushort flags = Le16(first, 6);
        ushort method = Le16(first, 8);
        if ((flags & 0b1001) != 0 || (method != 0 && method != 8))
        {
            return false;
        }

        uint compressed = Le32(first, 18);
        int nameLen = Le16(first, 26);
        int extraLen = Le16(first, 28);
        if (nameLen + 1 >= FirstEntryBytes)
        {
            return false;
        }

        if (first.Length < LocalHeaderLen + nameLen)
        {
            return false;
        }

        name = first.Slice(LocalHeaderLen, nameLen);
        int dataStart = LocalHeaderLen + nameLen + extraLen;
        if ((uint)dataStart > (uint)first.Length || first.Length - dataStart < compressed)
        {
            return false;
        }

        data = first.Slice(dataStart, (int)compressed);
        deflated = method == 8;
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
            if (!tail.Slice(at).StartsWith(EocdSignature))
            {
                continue;
            }

            ushort commentLen = BinaryPrimitives.ReadUInt16LittleEndian(tail.Slice(at + 20));
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
        if (!tail.Slice(locator).StartsWith(Zip64LocatorSignature))
        {
            return false;
        }

        ulong recordAbs = BinaryPrimitives.ReadUInt64LittleEndian(tail.Slice(locator + 8));
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

        entries = BinaryPrimitives.ReadUInt64LittleEndian(record.Slice(32));
        cdSize = BinaryPrimitives.ReadUInt64LittleEndian(record.Slice(40));
        cdOffset = BinaryPrimitives.ReadUInt64LittleEndian(record.Slice(48));
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ushort Le16(ReadOnlySpan<byte> s, int off) =>
        BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(off));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Le32(ReadOnlySpan<byte> s, int off) =>
        BinaryPrimitives.ReadUInt32LittleEndian(s.Slice(off));

    private readonly struct Eocd
    {
        public ushort Entries { get; init; }
        public uint CdSize { get; init; }
        public uint CdOffset { get; init; }
        public byte Flags { get; init; }

        public static Eocd Decode(ReadOnlySpan<byte> tail, int eocd)
        {
            ReadOnlySpan<byte> record = tail.Slice(eocd, EocdLen);
            ushort disk = BinaryPrimitives.ReadUInt16LittleEndian(record.Slice(4));
            ushort cdDisk = BinaryPrimitives.ReadUInt16LittleEndian(record.Slice(6));
            ushort entriesDisk = BinaryPrimitives.ReadUInt16LittleEndian(record.Slice(8));
            ushort entries = BinaryPrimitives.ReadUInt16LittleEndian(record.Slice(10));
            uint cdSize = BinaryPrimitives.ReadUInt32LittleEndian(record.Slice(12));
            uint cdOffset = BinaryPrimitives.ReadUInt32LittleEndian(record.Slice(16));
            bool locator = eocd >= Zip64LocatorLen
                && tail.Slice(eocd - Zip64LocatorLen).StartsWith(Zip64LocatorSignature);

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
