using System.Runtime.CompilerServices;

namespace Magika.Resolvers;

/// <summary>ZIP container subtypes matching upstream <c>magic::zip</c> / apk_prefix / epub_prefix.</summary>
internal static class ZipMagic
{
    private static ReadOnlySpan<byte> LocalSig => "PK\u0003\u0004"u8;

    /// <summary>Prefix-only APK local-header detection (<c>taxonomy_apk</c>).</summary>
    internal static string? ApkPrefix(ReadOnlySpan<byte> prefix)
    {
        const int Flags = 6;
        const int Method = 8;
        const int CompSize = 18;
        const int UncompSize = 22;
        const int NameLen = 26;
        const int NameOffset = 30;
        const int MinLen = 41;

        if (prefix.Length < MinLen || !prefix.StartsWith(LocalSig))
        {
            return null;
        }

        ushort? flags = MagicHelpers.Le16(prefix, Flags);
        ushort? method = MagicHelpers.Le16(prefix, Method);
        uint? comp = MagicHelpers.Le32(prefix, CompSize);
        uint? uncomp = MagicHelpers.Le32(prefix, UncompSize);
        ushort? nameLen = MagicHelpers.Le16(prefix, NameLen);
        if (flags is null || method is null || comp is null || uncomp is null || nameLen is null)
        {
            return null;
        }

        if (flags.Value % 2 != 0)
        {
            return null;
        }

        if (method.Value is not (0 or 8))
        {
            return null;
        }

        if (!((comp.Value >= 1 && uncomp.Value >= 1) || (flags.Value & 0x08) != 0))
        {
            return null;
        }

        bool valid =
            (nameLen.Value == 11
                && MagicHelpers.EqAt(prefix, NameOffset, "classes.dex"u8)
                && HasLocalEntry(prefix, 41, 4047, "AndroidManifest.xml"u8))
            || (prefix.Length >= 49
                && nameLen.Value == 19
                && MagicHelpers.EqAt(prefix, NameOffset, "AndroidManifest.xml"u8)
                && (HasLocalEntry(prefix, 49, 4055, "classes.dex"u8)
                    || HasLocalEntry(prefix, 49, 4052, "resources.arsc"u8)));
        return valid ? "apk" : null;
    }

    /// <summary>EPUB OCF local-header detection in the 4 KiB prefix.</summary>
    internal static string? EpubPrefix(ReadOnlySpan<byte> prefix)
    {
        const int Flags = 6;
        const int Method = 8;
        const int CompSize = 18;
        const int UncompSize = 22;
        const int NameLen = 26;
        const int ExtraLen = 28;
        const int PayloadOffset = 30;
        const int DescriptorOffset = 58;
        const int MinLen = 58;

        ReadOnlySpan<byte> payload = "mimetypeapplication/epub+zip"u8;
        // PK\x07\x08 + CRC32(application/epub+zip)=0x2CAB616F + 20 + 20
        ReadOnlySpan<byte> descriptor = new byte[]
        {
            0x50, 0x4B, 0x07, 0x08,
            0x6F, 0x61, 0xAB, 0x2C,
            0x14, 0x00, 0x00, 0x00,
            0x14, 0x00, 0x00, 0x00,
        };

        if (prefix.Length < MinLen
            || !prefix.StartsWith(LocalSig)
            || !MagicHelpers.EqAt(prefix, PayloadOffset, payload)
            || MagicHelpers.Le16(prefix, Method) != 0
            || MagicHelpers.Le16(prefix, NameLen) != 8
            || MagicHelpers.Le16(prefix, ExtraLen) != 0)
        {
            return null;
        }

        ushort? flags = MagicHelpers.Le16(prefix, Flags);
        uint? comp = MagicHelpers.Le32(prefix, CompSize);
        uint? uncomp = MagicHelpers.Le32(prefix, UncompSize);
        if (flags is null || comp is null || uncomp is null || flags.Value % 2 != 0)
        {
            return null;
        }

        if (comp.Value == 20 && uncomp.Value == 20)
        {
            return "epub";
        }

        if (prefix.Length >= 74
            && flags.Value % 16 == 8
            && comp.Value == 0
            && uncomp.Value == 0
            && MagicHelpers.EqAt(prefix, DescriptorOffset, descriptor))
        {
            return "epub";
        }

        return null;
    }

    /// <summary>Full ZIP subtype cascade after optional apk/epub prefix hits.</summary>
    internal static string? Identify(ReadOnlySpan<byte> prefix, ulong size, ReadOnlySpan<byte> tail)
    {
        if (ApkPrefix(prefix) is string apk)
        {
            return apk;
        }

        if (EpubPrefix(prefix) is string epub)
        {
            return epub;
        }

        using ZipFacts.Analysis facts = ZipFacts.Analyze(prefix, size, tail);
        ReadOnlySpan<byte> names = facts.Names;
        ReadOnlySpan<byte> first = facts.FirstEntry;
        bool valid = facts.Valid;
        bool dirNames = facts.HasDirNames;

        // Full rules
        if (valid)
        {
            if (names.IndexOf("\nmodel.weights.h5\n"u8) >= 0 && names.IndexOf("\nmetadata.json\n"u8) >= 0)
            {
                return "keras";
            }

            if (names.IndexOf(".qgs\n"u8) >= 0)
            {
                return "qgis";
            }

            if (names.IndexOf("\nvisio/document.xml\n"u8) >= 0)
            {
                return "visio";
            }

            if (dirNames && names.IndexOf("\n[Content_Types].xml\n"u8) >= 0 && names.IndexOf("\nxl/workbook.xml\n"u8) >= 0)
            {
                return "xlsx";
            }

            if (names.IndexOf("\n3D/3dmodel.model\n"u8) >= 0)
            {
                return "3mf";
            }
        }

        // Partial central-directory rules
        if (dirNames
            && names.IndexOf("\nAndroidManifest.xml\n"u8) >= 0
            && (names.IndexOf("\nclasses.dex\n"u8) >= 0 || names.IndexOf("\nresources.arsc\n"u8) >= 0))
        {
            return "apk";
        }

        if (names.StartsWith("\nmimetype=application/epub+zip\n"u8)
            && names.IndexOf("\nMETA-INF/container.xml\n"u8) >= 0)
        {
            return "epub";
        }

        if (names.StartsWith("\nmimetype=application/vnd.oasis.opendocument.presentation\n"u8)
            && names.IndexOf("\ncontent.xml\n"u8) >= 0)
        {
            return "odp";
        }

        if (names.StartsWith("\nmimetype=application/vnd.oasis.opendocument.spreadsheet\n"u8)
            && names.IndexOf("\ncontent.xml\n"u8) >= 0)
        {
            return "ods";
        }

        if (names.StartsWith("\nmimetype=application/vnd.oasis.opendocument.text\n"u8)
            && names.IndexOf("\ncontent.xml\n"u8) >= 0)
        {
            return "odt";
        }

        if (dirNames
            && names.IndexOf("\nMETA-INF/MANIFEST.MF\n"u8) >= 0
            && names.IndexOf(".class\n"u8) >= 0)
        {
            return "jar";
        }

        if (valid && names.IndexOf("\ndoc.kml\n"u8) >= 0)
        {
            return "kmz";
        }

        if (valid && names.IndexOf("\nAppxManifest.xml\n"u8) >= 0 && names.IndexOf("\nAppxBlockMap.xml\n"u8) >= 0)
        {
            return "msix";
        }

        if (dirNames && names.IndexOf("\n[Content_Types].xml\n"u8) >= 0 && names.IndexOf("\nppt/presentation.xml\n"u8) >= 0)
        {
            return "pptx";
        }

        // First-entry rules
        if (first.StartsWith("[Content_Types].xml\n"u8))
        {
            if (first.IndexOf("application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"u8) >= 0
                || first.IndexOf("application/vnd.ms-word.document.macroEnabled.main+xml"u8) >= 0)
            {
                return "docx";
            }

            if (first.IndexOf("application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml"u8) >= 0
                || first.IndexOf("application/vnd.ms-word.template.macroEnabledTemplate.main+xml"u8) >= 0)
            {
                return "dotx";
            }

            if (first.IndexOf("application/vnd.ms-excel.sheet.binary.macroEnabled.main"u8) >= 0)
            {
                return "xlsb";
            }
        }

        return null;
    }

    private static bool HasLocalEntry(ReadOnlySpan<byte> prefix, int lo, int hi, ReadOnlySpan<byte> name)
    {
        const int NameOffset = 30;
        const int NameLenAt = 26;
        ushort nameLen = (ushort)name.Length;
        int searchFrom = 0;
        while (true)
        {
            int namePos = prefix.Slice(searchFrom).IndexOf(name);
            if (namePos < 0)
            {
                return false;
            }

            namePos += searchFrom;
            int hdrPos = namePos - NameOffset;
            if (hdrPos < 0)
            {
                searchFrom = namePos + 1;
                continue;
            }

            if (hdrPos > hi)
            {
                return false;
            }

            if (hdrPos >= lo
                && MagicHelpers.EqAt(prefix, hdrPos, LocalSig)
                && MagicHelpers.Le16(prefix, hdrPos + NameLenAt) == nameLen)
            {
                return true;
            }

            searchFrom = namePos + 1;
        }
    }
}
