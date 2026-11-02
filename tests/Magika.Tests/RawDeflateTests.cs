using System.IO.Compression;
using System.Text;
using Magika.Resolvers;
using Xunit;

namespace Magika.Tests;

public class RawDeflateTests
{
    [Fact]
    public void InflatesRawDeflateMatchingDeflateStream()
    {
        byte[] payload = Encoding.UTF8.GetBytes(
            """<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>""");
        byte[] compressed = CompressRaw(payload);

        Span<byte> dst = new byte[payload.Length];
        Assert.True(RawDeflate.TryInflate(compressed, dst, out int written));
        Assert.Equal(payload.Length, written);
        Assert.True(payload.AsSpan().SequenceEqual(dst));
    }

    [Fact]
    public void StopsWhenDestinationIsFull()
    {
        byte[] payload = Encoding.ASCII.GetBytes(new string('A', 1024));
        byte[] compressed = CompressRaw(payload);
        Span<byte> dst = new byte[64];
        Assert.True(RawDeflate.TryInflate(compressed, dst, out int written));
        Assert.Equal(64, written);
        Assert.True(dst.ToArray().AsSpan().SequenceEqual(payload.AsSpan(0, 64)));
    }

    [Fact]
    public void Docx_WithDeflatedContentTypes()
    {
        byte[] zip = BuildDeflatedZip(
            ("[Content_Types].xml", Encoding.ASCII.GetBytes(
                """<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>""")));
        Assert.Equal("docx", ContentResolver.Identify(zip.AsSpan(0, Math.Min(ContentResolver.PrefixLimit, zip.Length)), zip.Length, zip));
    }

    private static byte[] CompressRaw(byte[] payload)
    {
        using MemoryStream ms = new();
        using (DeflateStream ds = new(ms, CompressionLevel.SmallestSize, leaveOpen: true))
        {
            ds.Write(payload);
        }

        return ms.ToArray();
    }

    private static byte[] BuildDeflatedZip(params (string Name, byte[] Data)[] entries)
    {
        using MemoryStream ms = new();
        using (ZipArchive archive = new(ms, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach ((string name, byte[] data) in entries)
            {
                ZipArchiveEntry entry = archive.CreateEntry(name, CompressionLevel.SmallestSize);
                using Stream stream = entry.Open();
                stream.Write(data);
            }
        }

        return ms.ToArray();
    }
}
