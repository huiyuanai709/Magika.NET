using System.Buffers.Binary;
using System.Runtime.CompilerServices;

namespace Magika.Resolvers;

/// <summary>Bounded PE analysis matching upstream <c>resolvers::facts::pe</c>.</summary>
internal static class PeFacts
{
    private const ushort PeMaxSections = 96;
    private const int ELfanewAt = 0x3c;
    private const int CoffHeaderLen = 20;
    private const int SectionHeaderLen = 40;
    private const ushort Pe32Magic = 0x10b;
    private const ushort Pe32PlusMagic = 0x20b;
    private const ulong Pe32FixedLen = 96;
    private const ulong Pe32PlusFixedLen = 112;
    private const ulong DirectoryEntries = 16;
    private const ulong DirectoryCertificateTable = 4;
    private const ulong DirectoryClrRuntimeHeader = 14;
    private const ushort CharacteristicExecutableImage = 0x0002;
    private const ushort CharacteristicDll = 0x2000;

    private const byte FlagSectionsNotHeld = 1 << 0;
    private const byte FlagCertificateOutOfFile = 1 << 1;
    private const byte FlagUnknownMagic = 1 << 2;
    private const byte FlagSectionBeyondFile = 1 << 3;

    private static ReadOnlySpan<byte> Mz => "MZ"u8;
    private static ReadOnlySpan<byte> PeSignature => "PE\0\0"u8;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool Wants(ReadOnlySpan<byte> prefix) => prefix.StartsWith(Mz);

    internal struct Facts
    {
        public bool Valid;
        public byte Flags;
        public ushort Machine;
        public ushort Characteristics;
        public ushort Subsystem;
        public uint EntryPoint;
        public ushort DllCharacteristics;
        public ushort Sections;
        public ushort Magic;
        public bool Clr;
        public bool Signed;
        public ulong Overlay;
        public bool IsDll;
        public bool IsExecutableImage;
    }

    internal static Facts Analyze(ReadOnlySpan<byte> first, ulong size)
    {
        Facts facts = default;
        if (!TryLocate(first, out int coff))
        {
            return facts;
        }

        facts.Machine = Le16(first, coff);
        facts.Sections = Le16(first, coff + 2);
        ulong soh = Le16(first, coff + 16);
        facts.Characteristics = Le16(first, coff + 18);
        facts.IsDll = (facts.Characteristics & CharacteristicDll) != 0;
        facts.IsExecutableImage = (facts.Characteristics & CharacteristicExecutableImage) != 0;
        int opt = coff + CoffHeaderLen;
        facts.Magic = Le16(first, opt);
        facts.EntryPoint = Le32(first, opt + 16);
        facts.Subsystem = Le16(first, opt + 68);
        facts.DllCharacteristics = Le16(first, opt + 70);

        ulong? fixedLen = facts.Magic switch
        {
            Pe32Magic => Pe32FixedLen,
            Pe32PlusMagic => Pe32PlusFixedLen,
            _ => null,
        };

        if (fixedLen is ulong fixedBytes)
        {
            ReadDirectories(first, size, opt, soh, fixedBytes, ref facts);
        }
        else
        {
            facts.Flags |= FlagUnknownMagic;
        }

        ulong table = (ulong)opt + (ulong)soh;
        ulong tableEnd = table + (ulong)SectionHeaderLen * facts.Sections;
        ulong rawEnd = tableEnd;
        if (tableEnd > (ulong)first.Length)
        {
            facts.Flags |= FlagSectionsNotHeld;
        }
        else
        {
            for (int index = 0; index < facts.Sections; index++)
            {
                int header = (int)table + SectionHeaderLen * index;
                ulong rawSize = Le32(first, header + 16);
                ulong rawPointer = Le32(first, header + 20);
                ulong extent = rawPointer + rawSize;
                if (extent > size)
                {
                    facts.Flags |= FlagSectionBeyondFile;
                }
                else if (extent > rawEnd)
                {
                    rawEnd = extent;
                }
            }
        }

        facts.Valid = fixedLen is ulong fl
            && soh >= fl
            && facts.Sections is >= 1 and <= PeMaxSections
            && (facts.Flags & FlagSectionsNotHeld) == 0
            && tableEnd <= size;
        if (facts.Valid)
        {
            facts.Overlay = size > rawEnd ? size - rawEnd : 0;
        }

        return facts;
    }

    /// <summary>Upstream <c>Facts::is_windows_pebin</c>.</summary>
    internal static bool IsWindowsPebin(in Facts p) =>
        p.Valid
        && p.IsExecutableImage
        && !(p.IsDll && p.EntryPoint == 0)
        && p.Subsystem is >= 1 and <= 16
        && p.Machine is 0x014C or 0x01C0 or 0x01C2 or 0x01C4 or 0x0200 or 0x5032 or 0x5064 or 0x8664 or 0xAA64;

    private static void ReadDirectories(ReadOnlySpan<byte> first, ulong size, int opt, ulong soh, ulong fixedLen, ref Facts facts)
    {
        ulong count = Le32(first, opt + (int)fixedLen - 4);
        if (TryDirectoryEntry(first, opt, soh, fixedLen, count, DirectoryCertificateTable, out uint offset, out uint length)
            && length != 0)
        {
            if ((ulong)offset + length > size)
            {
                facts.Flags |= FlagCertificateOutOfFile;
            }
            else
            {
                facts.Signed = offset != 0;
            }
        }

        if (TryDirectoryEntry(first, opt, soh, fixedLen, count, DirectoryClrRuntimeHeader, out uint rva, out uint clrLen))
        {
            facts.Clr = rva != 0 && clrLen != 0;
        }
    }

    private static bool TryDirectoryEntry(
        ReadOnlySpan<byte> first,
        int opt,
        ulong soh,
        ulong fixedLen,
        ulong count,
        ulong index,
        out uint rva,
        out uint size)
    {
        rva = 0;
        size = 0;
        if (index >= Math.Min(count, DirectoryEntries) || fixedLen + 8 * (index + 1) > soh)
        {
            return false;
        }

        int at = opt + (int)(fixedLen + 8 * index);
        if ((uint)(at + 8) > (uint)first.Length)
        {
            return false;
        }

        rva = Le32(first, at);
        size = Le32(first, at + 4);
        return true;
    }

    private static bool TryLocate(ReadOnlySpan<byte> first, out int coff)
    {
        coff = 0;
        if (!Wants(first))
        {
            return false;
        }

        if (first.Length < ELfanewAt + 4)
        {
            return false;
        }

        uint eLfanew = Le32(first, ELfanewAt);
        ulong headersEnd = eLfanew + (ulong)(PeSignature.Length + CoffHeaderLen);
        if (eLfanew < 4 || headersEnd > (ulong)first.Length)
        {
            return false;
        }

        int signature = (int)eLfanew;
        if (!first.Slice(signature, PeSignature.Length).SequenceEqual(PeSignature))
        {
            return false;
        }

        coff = signature + PeSignature.Length;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ushort Le16(ReadOnlySpan<byte> s, int off) =>
        (uint)(off + 2) <= (uint)s.Length ? BinaryPrimitives.ReadUInt16LittleEndian(s.Slice(off)) : (ushort)0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Le32(ReadOnlySpan<byte> s, int off) =>
        (uint)(off + 4) <= (uint)s.Length ? BinaryPrimitives.ReadUInt32LittleEndian(s.Slice(off)) : 0u;
}
