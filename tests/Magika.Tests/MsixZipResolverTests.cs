using System.IO.Compression;
using System.Text;
using Xunit;

namespace Magika.Tests;

public class MsixZipResolverTests
{
    [Fact]
    public void Identify_PlainZip_StaysZip()
    {
        MagikaSession session = MagikaSession.Create();
        byte[] zip = BuildZip(("hello.txt", Encoding.ASCII.GetBytes("hi")));
        MagikaResult result = session.Identify(zip);
        Assert.Equal("zip", result.ContentType.Label);
        Assert.Null(result.OverwriteReason);
    }

    [Fact]
    public void Identify_ZipWithBothAppxFiles_IsMsix()
    {
        MagikaSession session = MagikaSession.Create();
        byte[] zip = BuildZip(
            ("AppxManifest.xml", Encoding.UTF8.GetBytes("<Package />")),
            ("AppxBlockMap.xml", Encoding.UTF8.GetBytes("<BlockMap />")),
            ("Assets/logo.png", new byte[] { 0x89, 0x50, 0x4E, 0x47 }));

        MagikaResult result = session.Identify(zip);
        Assert.Equal("msix", result.ContentType.Label);
        Assert.Equal("application/msix", result.ContentType.MimeType);
        Assert.Equal(OverwriteReason.RulesVeto, result.OverwriteReason);
        Assert.Null(result.ModelContentType);
    }

    [Fact]
    public void Identify_ZipWithOnlyAppxManifest_IsNotMsix()
    {
        MagikaSession session = MagikaSession.Create();
        byte[] zip = BuildZip(
            ("AppxManifest.xml", Encoding.UTF8.GetBytes("<Package />")),
            ("other.bin", new byte[] { 1, 2, 3 }));

        MagikaResult result = session.Identify(zip);
        Assert.NotEqual("msix", result.ContentType.Label);
        Assert.Equal("zip", result.ContentType.Label);
    }

    [Fact]
    public void Identify_ZipWithOnlyAppxBlockMap_IsNotMsix()
    {
        MagikaSession session = MagikaSession.Create();
        byte[] zip = BuildZip(
            ("AppxBlockMap.xml", Encoding.UTF8.GetBytes("<BlockMap />")),
            ("other.bin", new byte[] { 1, 2, 3 }));

        MagikaResult result = session.Identify(zip);
        Assert.NotEqual("msix", result.ContentType.Label);
        Assert.Equal("zip", result.ContentType.Label);
    }

    [Fact]
    public void Identify_NestedAppxNames_AreNotMsix()
    {
        MagikaSession session = MagikaSession.Create();
        byte[] zip = BuildZip(
            ("Payload/AppxManifest.xml", Encoding.UTF8.GetBytes("<Package />")),
            ("Payload/AppxBlockMap.xml", Encoding.UTF8.GetBytes("<BlockMap />")));

        MagikaResult result = session.Identify(zip);
        Assert.Equal("zip", result.ContentType.Label);
    }

    [Fact]
    public void IdentifyPath_MsixLikeZip_IsMsix()
    {
        MagikaSession session = MagikaSession.Create();
        string path = Path.Combine(Path.GetTempPath(), "magika-msix-" + Guid.NewGuid().ToString("n") + ".zip");
        try
        {
            File.WriteAllBytes(
                path,
                BuildZip(
                    ("AppxManifest.xml", Encoding.UTF8.GetBytes("<Package />")),
                    ("AppxBlockMap.xml", Encoding.UTF8.GetBytes("<BlockMap />"))));

            MagikaResult result = session.IdentifyPath(path);
            Assert.Equal("msix", result.ContentType.Label);
            Assert.Equal(OverwriteReason.RulesVeto, result.OverwriteReason);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ZipMsixResolver_RequiresValidCentralDirectory()
    {
        byte[] zip = BuildZip(
            ("AppxManifest.xml", Encoding.UTF8.GetBytes("<Package />")),
            ("AppxBlockMap.xml", Encoding.UTF8.GetBytes("<BlockMap />")));

        Assert.True(ZipMsixResolver.WantsTail(zip));
        Assert.True(ZipMsixResolver.IsMsix(zip, zip.Length, zip));

        byte[] truncated = zip[..^4];
        Assert.False(ZipMsixResolver.IsMsix(truncated.AsSpan(0, Math.Min(30, truncated.Length)), truncated.Length, truncated));
    }

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
}
