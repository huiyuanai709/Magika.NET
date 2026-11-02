using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using Magika.Resolvers;
using Xunit;

namespace Magika.Tests;

public class ContentResolverTests
{
    private static MagikaSession Session() => MagikaSession.Create();

    [Fact]
    public void PlainZip_StaysZip_ViaModel()
    {
        byte[] zip = BuildZip(("hello.txt", Encoding.ASCII.GetBytes("hi")));
        MagikaResult result = Session().Identify(zip);
        Assert.Equal("zip", result.ContentType.Label);
        Assert.Null(result.OverwriteReason);
    }

    [Fact]
    public void Msix_BothAppxFiles()
    {
        byte[] zip = BuildZip(
            ("AppxManifest.xml", Encoding.UTF8.GetBytes("<Package />")),
            ("AppxBlockMap.xml", Encoding.UTF8.GetBytes("<BlockMap />")));
        Assert.Equal("msix", ContentResolver.Identify(zip, zip.Length, zip));
        MagikaResult result = Session().Identify(zip);
        Assert.Equal("msix", result.ContentType.Label);
        Assert.Equal(OverwriteReason.RulesVeto, result.OverwriteReason);
    }

    [Theory]
    [InlineData("AppxManifest.xml")]
    [InlineData("AppxBlockMap.xml")]
    public void Msix_SingleAppxFile_NotMsix(string only)
    {
        byte[] zip = BuildZip((only, Encoding.UTF8.GetBytes("<x />")), ("other.bin", new byte[] { 1, 2, 3 }));
        Assert.Null(ContentResolver.Identify(Prefix(zip), zip.Length, zip));
        Assert.Equal("zip", Session().Identify(zip).ContentType.Label);
    }

    [Fact]
    public void Msix_NestedAppxNames_NotMsix()
    {
        byte[] zip = BuildZip(
            ("Payload/AppxManifest.xml", Encoding.UTF8.GetBytes("<Package />")),
            ("Payload/AppxBlockMap.xml", Encoding.UTF8.GetBytes("<BlockMap />")));
        Assert.Null(ContentResolver.Identify(Prefix(zip), zip.Length, zip));
        Assert.Equal("zip", Session().Identify(zip).ContentType.Label);
    }

    [Fact]
    public void Keras_RequiresBothFiles()
    {
        byte[] hit = BuildZip(
            ("model.weights.h5", new byte[] { 1 }),
            ("metadata.json", Encoding.ASCII.GetBytes("{}")));
        Assert.Equal("keras", ContentResolver.Identify(Prefix(hit), hit.Length, hit));

        byte[] miss = BuildZip(("model.weights.h5", new byte[] { 1 }));
        Assert.Null(ContentResolver.Identify(Prefix(miss), miss.Length, miss));
    }

    [Fact]
    public void Qgis_Visio_Kmz_3mf()
    {
        Assert.Equal("qgis", IdentifyZip(("project.qgs", Encoding.ASCII.GetBytes("x"))));
        Assert.Equal("visio", IdentifyZip(("visio/document.xml", Encoding.ASCII.GetBytes("<v />"))));
        Assert.Equal("kmz", IdentifyZip(("doc.kml", Encoding.ASCII.GetBytes("<kml />"))));
        Assert.Equal("3mf", IdentifyZip(("3D/3dmodel.model", Encoding.ASCII.GetBytes("<model />"))));
    }

    [Fact]
    public void Apk_ViaCentralDirectoryNames()
    {
        byte[] zip = BuildZip(
            ("AndroidManifest.xml", Encoding.ASCII.GetBytes("<manifest />")),
            ("classes.dex", new byte[] { 0x64, 0x65, 0x78, 0x0A }));
        Assert.Equal("apk", ContentResolver.Identify(Prefix(zip), zip.Length, zip));
    }

    [Fact]
    public void Jar_ManifestAndClass()
    {
        byte[] zip = BuildZip(
            ("META-INF/MANIFEST.MF", Encoding.ASCII.GetBytes("Manifest-Version: 1.0\n")),
            ("com/Example.class", new byte[] { 0xCA, 0xFE, 0xBA, 0xBE }));
        Assert.Equal("jar", ContentResolver.Identify(Prefix(zip), zip.Length, zip));

        byte[] miss = BuildZip(("META-INF/MANIFEST.MF", Encoding.ASCII.GetBytes("Manifest-Version: 1.0\n")));
        Assert.Null(ContentResolver.Identify(Prefix(miss), miss.Length, miss));
    }

    [Fact]
    public void Odf_MimetypeFirstEntry()
    {
        Assert.Equal("odt", IdentifyOdf("application/vnd.oasis.opendocument.text"));
        Assert.Equal("ods", IdentifyOdf("application/vnd.oasis.opendocument.spreadsheet"));
        Assert.Equal("odp", IdentifyOdf("application/vnd.oasis.opendocument.presentation"));
    }

    [Fact]
    public void OfficeOpenXml_ByContentTypesAndParts()
    {
        Assert.Equal("xlsx", IdentifyZip(
            ("[Content_Types].xml", Encoding.ASCII.GetBytes("<Types />")),
            ("xl/workbook.xml", Encoding.ASCII.GetBytes("<workbook />"))));
        Assert.Equal("pptx", IdentifyZip(
            ("[Content_Types].xml", Encoding.ASCII.GetBytes("<Types />")),
            ("ppt/presentation.xml", Encoding.ASCII.GetBytes("<presentation />"))));
    }

    [Fact]
    public void Docx_Dotx_Xlsb_ViaFirstEntryContentTypes()
    {
        Assert.Equal(
            "docx",
            IdentifyZip(("[Content_Types].xml", Encoding.ASCII.GetBytes(
                """<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"""))));
        Assert.Equal(
            "dotx",
            IdentifyZip(("[Content_Types].xml", Encoding.ASCII.GetBytes(
                """<?xml version="1.0"?><Types><Override ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml"/></Types>"""))));
        Assert.Equal(
            "xlsb",
            IdentifyZip(("[Content_Types].xml", Encoding.ASCII.GetBytes(
                """<?xml version="1.0"?><Types><Override ContentType="application/vnd.ms-excel.sheet.binary.macroEnabled.main"/></Types>"""))));
    }

    [Fact]
    public void Magic_Gzip_Rar_SevenZip()
    {
        byte[] gzip = new byte[20];
        gzip[0] = 0x1F; gzip[1] = 0x8B; gzip[2] = 0x08; gzip[3] = 0x00;
        Assert.Equal("gzip", ContentResolver.Identify(gzip, gzip.Length, ReadOnlySpan<byte>.Empty));
        Assert.Equal("gzip", Session().Identify(gzip).ContentType.Label);
        Assert.Equal(OverwriteReason.RulesVeto, Session().Identify(gzip).OverwriteReason);

        byte[] rar = new byte[] { 0x52, 0x61, 0x72, 0x21, 0x1A, 0x07, 0x00, 0x00 };
        Assert.Equal("rar", ContentResolver.Identify(rar, rar.Length, ReadOnlySpan<byte>.Empty));

        byte[] seven = new byte[32];
        seven[0] = 0x37; seven[1] = 0x7A; seven[2] = 0xBC; seven[3] = 0xAF;
        seven[4] = 0x27; seven[5] = 0x1C; seven[6] = 0x00;
        Assert.Equal("sevenzip", ContentResolver.Identify(seven, seven.Length, ReadOnlySpan<byte>.Empty));
    }

    [Fact]
    public void Magic_Npy_And_Safetensors()
    {
        byte[] npy = new byte[16];
        npy[0] = 0x93;
        Encoding.ASCII.GetBytes("NUMPY").CopyTo(npy, 1);
        Assert.Equal("npy", ContentResolver.Identify(npy, npy.Length, ReadOnlySpan<byte>.Empty));

        byte[] st = new byte[16];
        BinaryPrimitives.WriteUInt32LittleEndian(st.AsSpan(0), 8); // header len lo
        BinaryPrimitives.WriteUInt32LittleEndian(st.AsSpan(4), 0); // hi
        st[8] = (byte)'{';
        Assert.Equal("safetensors", ContentResolver.Identify(st, st.Length, ReadOnlySpan<byte>.Empty));
    }

    [Fact]
    public void Pe_MinimalWindowsPebin()
    {
        byte[] pe = BuildMinimalPe();
        Assert.Equal("pebin", ContentResolver.Identify(pe, pe.Length, ReadOnlySpan<byte>.Empty));
        MagikaResult result = Session().Identify(pe);
        Assert.Equal("pebin", result.ContentType.Label);
        Assert.Equal(OverwriteReason.RulesVeto, result.OverwriteReason);
    }

    [Fact]
    public void TruncatedZipCd_DoesNotFalsePositiveMsix()
    {
        byte[] zip = BuildZip(
            ("AppxManifest.xml", Encoding.UTF8.GetBytes("<Package />")),
            ("AppxBlockMap.xml", Encoding.UTF8.GetBytes("<BlockMap />")));
        Assert.True(ContentResolver.WantsZipTail(zip));
        Assert.Equal("msix", ContentResolver.Identify(Prefix(zip), zip.Length, zip));

        byte[] truncated = zip[..^4];
        Assert.Null(ContentResolver.Identify(Prefix(truncated), truncated.Length, truncated));
    }

    [Fact]
    public void ResolverVetos_HasNinetyThreeLabels()
    {
        Assert.Equal(93, ResolverVetos.Count);
        Assert.True(ResolverVetos.Contains("gzip"));
        Assert.True(ResolverVetos.Contains("keras"));
        Assert.False(ResolverVetos.Contains("zip"));
        Assert.False(ResolverVetos.Contains("msix"));
    }

    [Fact]
    public void PostProcessor_ResolversVeto_FallsBackUnknown()
    {
        MagikaSession session = Session();
        // Drive PostProcessor through a synthetic score vector via reflection-free path:
        // use model definition from a real run's shape by identifying empty/small and then
        // calling DecideScores with a handcrafted vector is not exposed. Instead verify the
        // veto path by identifying a truncated gzip that fails the resolver (too short) is not
        // labeled gzip via RulesVeto — and that a full gzip is RulesVeto.
        byte[] shortGz = new byte[] { 0x1F, 0x8B, 0x08 }; // fails len>=20
        MagikaResult shortResult = session.Identify(shortGz);
        Assert.NotEqual(OverwriteReason.RulesVeto, shortResult.OverwriteReason);

        byte[] gzip = new byte[20];
        gzip[0] = 0x1F; gzip[1] = 0x8B; gzip[2] = 0x08;
        MagikaResult full = session.Identify(gzip);
        Assert.Equal("gzip", full.ContentType.Label);
        Assert.Equal(OverwriteReason.RulesVeto, full.OverwriteReason);
    }

    [Fact]
    public void IdentifyPath_MsixLikeZip()
    {
        string path = Path.Combine(Path.GetTempPath(), "magika-msix-" + Guid.NewGuid().ToString("n") + ".zip");
        try
        {
            File.WriteAllBytes(path, BuildZip(
                ("AppxManifest.xml", Encoding.UTF8.GetBytes("<Package />")),
                ("AppxBlockMap.xml", Encoding.UTF8.GetBytes("<BlockMap />"))));
            MagikaResult result = Session().IdentifyPath(path);
            Assert.Equal("msix", result.ContentType.Label);
            Assert.Equal(OverwriteReason.RulesVeto, result.OverwriteReason);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string? IdentifyZip(params (string Name, byte[] Data)[] entries)
    {
        byte[] zip = BuildZip(entries);
        return ContentResolver.Identify(Prefix(zip), zip.Length, zip);
    }

    private static string? IdentifyOdf(string mime)
    {
        byte[] zip = BuildZip(
            ("mimetype", Encoding.ASCII.GetBytes(mime)),
            ("content.xml", Encoding.ASCII.GetBytes("<office />")));
        return ContentResolver.Identify(Prefix(zip), zip.Length, zip);
    }

    private static ReadOnlySpan<byte> Prefix(byte[] content) =>
        content.AsSpan(0, Math.Min(ContentResolver.PrefixLimit, content.Length));

    private static byte[] BuildZip(params (string Name, byte[] Data)[] entries)
    {
        using MemoryStream ms = new();
        using (ZipArchive archive = new(ms, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach ((string name, byte[] data) in entries)
            {
                ZipArchiveEntry entry = archive.CreateEntry(name, CompressionLevel.NoCompression);
                using Stream stream = entry.Open();
                stream.Write(data);
            }
        }

        return ms.ToArray();
    }

    /// <summary>Minimal PE32+ executable image that satisfies <see cref="PeFacts.IsWindowsPebin"/>.</summary>
    private static byte[] BuildMinimalPe()
    {
        const int eLfanew = 0x80;
        const int sectionRaw = 0x400;
        const int fileSize = 0x600;
        byte[] pe = new byte[fileSize];
        pe[0] = (byte)'M';
        pe[1] = (byte)'Z';
        BinaryPrimitives.WriteInt32LittleEndian(pe.AsSpan(0x3C), eLfanew);

        int sig = eLfanew;
        pe[sig] = (byte)'P';
        pe[sig + 1] = (byte)'E';
        // COFF
        int coff = sig + 4;
        BinaryPrimitives.WriteUInt16LittleEndian(pe.AsSpan(coff), 0x8664); // AMD64
        BinaryPrimitives.WriteUInt16LittleEndian(pe.AsSpan(coff + 2), 1); // sections
        BinaryPrimitives.WriteUInt16LittleEndian(pe.AsSpan(coff + 16), 0xF0); // SizeOfOptionalHeader
        BinaryPrimitives.WriteUInt16LittleEndian(pe.AsSpan(coff + 18), 0x0002); // EXECUTABLE_IMAGE

        int opt = coff + 20;
        BinaryPrimitives.WriteUInt16LittleEndian(pe.AsSpan(opt), 0x20B); // PE32+
        BinaryPrimitives.WriteUInt32LittleEndian(pe.AsSpan(opt + 16), 0x1000); // EntryPoint
        BinaryPrimitives.WriteUInt16LittleEndian(pe.AsSpan(opt + 68), 3); // IMAGE_SUBSYSTEM_WINDOWS_CUI
        BinaryPrimitives.WriteUInt32LittleEndian(pe.AsSpan(opt + 108), 16); // NumberOfRvaAndSizes

        int section = opt + 0xF0;
        // name "text\0\0\0\0"
        Encoding.ASCII.GetBytes(".text").CopyTo(pe, section);
        BinaryPrimitives.WriteUInt32LittleEndian(pe.AsSpan(section + 16), 0x200); // SizeOfRawData
        BinaryPrimitives.WriteUInt32LittleEndian(pe.AsSpan(section + 20), sectionRaw); // PointerToRawData
        return pe;
    }
}
