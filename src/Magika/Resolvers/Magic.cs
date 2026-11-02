#nullable enable
using System.Runtime.CompilerServices;

namespace Magika.Resolvers;

/// <summary>
/// Magic-keyed dispatch cascade ported from google/magika <c>resolvers/magic/mod.rs</c>.
/// ZIP subtypes: <see cref="ZipMagic"/>; PE: <see cref="PeFacts"/>.
/// </summary>
internal static class Magic
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static string? Identify(in MagicCtx ctx)
    {
        ReadOnlySpan<byte> prefix = ctx.Prefix;
        if (prefix.Length >= 4)
        {
            if (Magic32(prefix, in ctx) is string m32) return m32;
        }
        if (prefix.Length >= 2)
        {
            if (Magic16(prefix, in ctx) is string m16) return m16;
        }
        if (OffsetProbes(prefix) is string off) return off;
        return Heuristics(prefix, in ctx);
    }

    private static string? Magic32(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
    {
        if (prefix.StartsWith("3DSX"u8))
        {
            if (ThreeDsx(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("bpli"u8))
        {
            if (AppleBplist(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x00 && prefix[1] == 0x05 && prefix[2] == 0x16 && prefix[3] == 0x07))
        {
            if (AppleDouble(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x00 && prefix[1] == 0x05 && prefix[2] == 0x16 && prefix[3] == 0x00))
        {
            if (AppleSingle(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(".snd"u8) || prefix.StartsWith("dns."u8))
        {
            if (Au(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("RIFF"u8) || prefix.StartsWith("RIFX"u8) || prefix.StartsWith("RF64"u8))
        {
            if (Riff(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x4F, 0x62, 0x6A, 0x01 }))
        {
            if (Avro(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x42, 0x41, 0x4D, 0x01 }))
        {
            if (Bam(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x0F && prefix[1] == 0x37 && prefix[2] == 0x42 && prefix[3] == 0x45) || prefix.StartsWith("FOR1"u8))
        {
            if (Beam(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("BLEN"u8))
        {
            if (Blend(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x42, 0x50, 0x47, 0xFB }))
        {
            if (Bpg(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("CRAM"u8))
        {
            if (Cram(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x7F, 0x45, 0x4C, 0x46 }))
        {
            if (Cubin(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x64, 0x65, 0x78, 0x0A }))
        {
            if (Dex(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x00 && prefix[1] == 0x00 && prefix[2] == 0x00 && prefix[3] == 0x01))
        {
            if (Dsstore(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("AC10"u8))
        {
            if (Dwg(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("Kayd"u8) || prefix.StartsWith("; FB"u8))
        {
            if (Fbx(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("SIMP"u8))
        {
            if (Fits(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x46, 0x4C, 0x56, 0x01 }))
        {
            if (Flv(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("SQLi"u8))
        {
            if (GeoPackage(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("GGUF"u8))
        {
            if (Gguf(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("glTF"u8))
        {
            if (Glb(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("GRIB"u8))
        {
            if (Grib(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x0E && prefix[1] == 0x03 && prefix[2] == 0x13 && prefix[3] == 0x01))
        {
            if (Hdf4(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("regf"u8))
        {
            if (Hve(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("icns"u8))
        {
            if (Icns(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x8B, 0x4A, 0x4E, 0x47 }))
        {
            if (Jng(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x42, 0x43, 0xC0, 0xDE }) || (prefix[0] == 0xDE && prefix[1] == 0xC0 && prefix[2] == 0x17 && prefix[3] == 0x0B))
        {
            if (LlvmBitcode(prefix, in ctx) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x4C && prefix[1] == 0x00 && prefix[2] == 0x00 && prefix[3] == 0x00))
        {
            if (Lnk(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("LRZI"u8))
        {
            if (Lrz(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x1B, 0x4C, 0x75, 0x61 }) || prefix.StartsWith(new byte[] { 0x1B, 0x4C, 0x4A, 0x01 }) || prefix.StartsWith(new byte[] { 0x1B, 0x4C, 0x4A, 0x02 }))
        {
            if (LuaBytecode(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("LZIP"u8))
        {
            if (Lzip(prefix, in ctx) is string hit) return hit;
            return null;
        }
        if (((prefix[0] == 0x02 || prefix[0] == 0x03) && prefix[1] == 0x21 && prefix[2] == 0x4C && prefix[3] == 0x18) || (prefix[0] == 0x04 && prefix[1] == 0x22 && prefix[2] == 0x4D && prefix[3] == 0x18))
        {
            if (Lz4(prefix, in ctx) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("MATL"u8))
        {
            if (Mat(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("MThd"u8))
        {
            if (Midi(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("MDMP"u8))
        {
            if (MiniDump(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x93, 0x4E, 0x55, 0x4D }))
        {
            if (Npy(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("NRRD"u8))
        {
            if (Nrrd(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0xE4 && prefix[1] == 0x52 && prefix[2] == 0x5C && prefix[3] == 0x7B))
        {
            if (One(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("PAR1"u8))
        {
            if (Parquet(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("----"u8))
        {
            if (PemOrPgp(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x70, 0x6C, 0x79, 0x0A }) || prefix.StartsWith(new byte[] { 0x70, 0x6C, 0x79, 0x0D }))
        {
            if (Ply(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("PGDM"u8))
        {
            if (PostgresDump(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("8BPS"u8))
        {
            if (Psd(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("qoif"u8))
        {
            if (Qoi(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("Rar!"u8) || prefix.StartsWith("RE~^"u8))
        {
            if (Rar(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("REDI"u8))
        {
            if (RedisRdb(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("RZIP"u8))
        {
            if (Rzip(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x37 && prefix[1] == 0x7A && prefix[2] == 0xBC && prefix[3] == 0xAF))
        {
            if (SevenZip(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x00 && prefix[1] == 0x00 && prefix[2] == 0x27 && prefix[3] == 0x0A))
        {
            if (ShapeFile(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0xFF && prefix[1] == 0xFE && prefix[2] == 0xFF && prefix[3] == 0x0E) || prefix.StartsWith(new byte[] { 0x0E, 0x53, 0x6B, 0x65 }))
        {
            if (SketchUp(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x03 && prefix[1] == 0x02 && prefix[2] == 0x23 && prefix[3] == 0x07) || (prefix[0] == 0x07 && prefix[1] == 0x23 && prefix[2] == 0x02 && prefix[3] == 0x03))
        {
            if (Spirv(prefix, in ctx) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("$FL2"u8) || prefix.StartsWith("$FL3"u8) || (prefix[0] == 0xC9 && prefix[1] == 0xC3 && prefix[2] == 0xE2 && prefix[3] == 0xC1))
        {
            if (Spss(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("ISO-"u8))
        {
            if (Step(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x55, 0x46, 0x32, 0x0A }))
        {
            if (Uf2(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("cone"u8))
        {
            if (Vhd(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("!<ar"u8))
        {
            if (Vib(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("IWAD"u8) || prefix.StartsWith("PWAD"u8))
        {
            if (Wad(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x00, 0x61, 0x73, 0x6D }))
        {
            if (Wasm(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("wOFF"u8))
        {
            if (Woff(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("wOF2"u8))
        {
            if (Woff2(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("xar!"u8))
        {
            if (Xar(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("gimp"u8))
        {
            if (Xcf(prefix) is string hit) return hit;
            return null;
        }
        if (((prefix[0] >= 0x22 && prefix[0] <= 0x28) && prefix[1] == 0xB5 && prefix[2] == 0x2F && prefix[3] == 0xFD))
        {
            if (Zst(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("ARRO"u8))
        {
            if (Arrow(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x0A, 0x0D, 0x0D, 0x0A }))
        {
            if (PcapNg(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("MSCF"u8))
        {
            if (Cab(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("Cr24"u8))
        {
            if (Crx(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("fLaC"u8))
        {
            if (Flac(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("GIF8"u8))
        {
            if (Gif(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x3F && prefix[1] == 0x5F && prefix[2] == 0x03 && prefix[3] == 0x00))
        {
            if (Hlp(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x00 && prefix[1] == 0x00 && prefix[2] == 0x01 && prefix[3] == 0x00))
        {
            if (Ico(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("FORM"u8))
        {
            if (Ilbm(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x00 && prefix[1] == 0x00 && prefix[2] == 0x00 && prefix[3] == 0x0C))
        {
            if (Jp2OrJxl(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0xFF && prefix[1] == 0x80 && prefix[2] == 0xC9 && prefix[3] == 0xC7))
        {
            if (Koala(prefix) is string hit) return hit;
            return null;
        }
        if (((prefix[0] == 0xCE || prefix[0] == 0xCF) && prefix[1] == 0xFA && prefix[2] == 0xED && prefix[3] == 0xFE) || (prefix[0] == 0xFE && prefix[1] == 0xED && prefix[2] == 0xFA && (prefix[3] == 0xCE || prefix[3] == 0xCF)))
        {
            if (Macho(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x1A && prefix[1] == 0x45 && prefix[2] == 0xDF && prefix[3] == 0xA3))
        {
            if (Mkv(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("SZDD"u8))
        {
            if (MsCompress(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x43, 0x44, 0x46, 0x01 }) || prefix.StartsWith(new byte[] { 0x43, 0x44, 0x46, 0x02 }))
        {
            if (NetCdf(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("OggS"u8))
        {
            if (Ogg(prefix) is string hit) return hit;
            return null;
        }
        if (false || (prefix[0] == 0xA1 && prefix[1] == 0xB2 && prefix[2] == 0x3C && prefix[3] == 0x4D))
        {
            if (Pcap(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("Micr"u8))
        {
            if (Pdb(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0xED && prefix[1] == 0xAB && prefix[2] == 0xEE && prefix[3] == 0xDB))
        {
            if (Rpm(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("<sta"u8))
        {
            if (Stata(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("%!PS"u8))
        {
            if (PostScript(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x00 && prefix[1] == 0x00 && prefix[2] == 0x00 && prefix[3] == 0x00))
        {
            if (Sas(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x01 && prefix[1] == 0x00 && prefix[2] == 0x00 && prefix[3] == 0x00))
        {
            if (Emf(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("d13:"u8) || prefix.StartsWith("d4:i"u8) || prefix.StartsWith("d8:a"u8) || prefix.StartsWith("d7:c"u8))
        {
            if (Torrent(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith(new byte[] { 0x50, 0x4B, 0x03, 0x04 }) || prefix.StartsWith(new byte[] { 0x50, 0x4B, 0x05, 0x06 }))
        {
            if (Zip(prefix, in ctx) is string hit) return hit;
            return null;
        }
        return null;
    }
    private static string? Magic16(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
    {
        if (prefix.StartsWith("MM"u8))
        {
            if (ThreeDsm(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("BZ"u8))
        {
            if (Bzip(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x1F && prefix[1] == 0x8B))
        {
            if (Gzip(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("P1"u8) || prefix.StartsWith("P2"u8) || prefix.StartsWith("P3"u8) || prefix.StartsWith("P4"u8) || prefix.StartsWith("P5"u8) || prefix.StartsWith("P6"u8))
        {
            if (Pbm(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("FW"u8) || prefix.StartsWith("CW"u8) || prefix.StartsWith("ZW"u8))
        {
            if (Swf(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x01 && (prefix[1] == 0xF7 || prefix[1] == 0xDF)))
        {
            if (Xcoff(prefix) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("BM"u8))
        {
            if (Bmp(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x1F && prefix[1] == 0x9D))
        {
            if (UnixCompress(prefix) is string hit) return hit;
            return null;
        }
        if ((prefix[0] == 0x00 && (prefix[1] >= 0x00 && prefix[1] <= 0x02)))
        {
            if (Degas(prefix, in ctx) is string hit) return hit;
            return null;
        }
        if (prefix.StartsWith("MZ"u8))
        {
            if (Pe(prefix, in ctx) is string hit) return hit;
            return null;
        }
        return null;
    }

    private static string? OffsetProbes(ReadOnlySpan<byte> prefix)
    {
        if (Ftyp(prefix) is string a) return a;
        if (Ace(prefix) is string b) return b;
        if (Dicom(prefix) is string c) return c;
        if (Duckdb(prefix) is string d) return d;
        if (Ese(prefix) is string e) return e;
        if (Icc(prefix) is string f) return f;
        if (Palmos(prefix) is string g) return g;
        if (OsmPbf(prefix) is string h) return h;
        return null;
    }


    private static string? Heuristics(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
    {
        var xml = XmlRoots(prefix);
        if (xml.Collada) return "collada";
        if (GltfJson(prefix) is string gltf) return gltf;
        if (xml.Gpx) return "gpx";
        if (xml.Kml) return "kml";
        if (Mpegts(prefix) is string ts) return ts;
        if (Safetensors(prefix) is string st) return st;
        if (Dbase(prefix, in ctx) is string db) return db;
        if (IntelHex(prefix) is string ih) return ih;
        if (xml.Osm) return "osm";
        return PythonBytecode(prefix);
    }

    private struct XmlRootFlags
    {
        public bool Collada;
        public bool Gpx;
        public bool Kml;
        public bool Osm;
    }

    private static XmlRootFlags XmlRoots(ReadOnlySpan<byte> prefix)
    {
        // Approximate upstream xml_roots: look for distinctive root tags in the prefix.
        XmlRootFlags f = default;
        if (prefix.IndexOf("<COLLADA"u8) >= 0 || prefix.IndexOf("<collada"u8) >= 0) f.Collada = true;
        if (prefix.IndexOf("<gpx"u8) >= 0 || prefix.IndexOf("<GPX"u8) >= 0) f.Gpx = true;
        if (prefix.IndexOf("<kml"u8) >= 0 || prefix.IndexOf("<KML"u8) >= 0) f.Kml = true;
        if (prefix.IndexOf("<osm"u8) >= 0 || prefix.IndexOf("<OSM"u8) >= 0) f.Osm = true;
        return f;
    }

    private static string? Ace(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 31) return null;
        if (!MagicHelpers.EqAt(prefix, 7, "**ACE**"u8)) return null;
        if (MagicHelpers.Le16(prefix, 2) is not ushort hdr || hdr < 27) return null;
        if (prefix[4] != 0) return null;
        if ((prefix[5] & 1) != 0) return null;
        return "ace";
    }


private static string? ApkPrefix(ReadOnlySpan<byte> prefix) => ZipMagic.ApkPrefix(prefix);

    private static string? AppleBplist(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("bplist"u8))) return null;
        return "applebplist";
    }

    private static string? AppleDouble(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (appledouble): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? AppleSingle(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (applesingle): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Arrow(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x41, 0x52, 0x52, 0x4F, 0x57, 0x31, 0x00, 0x00 }))) return null;
        return "arrow";
    }

    private static string? Au(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(".snd"u8) || prefix.StartsWith("dns."u8))) return null;
        return "au";
    }

    private static string? Avro(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x4F, 0x62, 0x6A, 0x01 }))) return null;
        return "avro";
    }

    private static string? Bam(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x42, 0x41, 0x4D, 0x01 }))) return null;
        return "bam";
    }

    private static string? Beam(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x0F, 0x37, 0x42, 0x45, 0x41, 0x4D, 0x21 }) || prefix.StartsWith("FOR1"u8))) return null;
        return "beam";
    }

    private static string? Blend(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("BLENDER"u8))) return null;
        return "blend";
    }

    private static string? Bmp(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 26 || !prefix.StartsWith("BM"u8)) return null;
        if (MagicHelpers.Le32(prefix, 6) != 0) return null;
        uint? fileSize = MagicHelpers.Le32(prefix, 2);
        uint? offBits = MagicHelpers.Le32(prefix, 10);
        uint? dib = MagicHelpers.Le32(prefix, 14);
        if (fileSize is null || offBits is null || dib is null) return null;
        if (fileSize.Value < 26 || offBits.Value < 26) return null;
        if (dib.Value == 12)
        {
            return MagicHelpers.Le16(prefix, 18) is ushort w && w >= 1
                && MagicHelpers.Le16(prefix, 20) is ushort h && h >= 1
                && MagicHelpers.Le16(prefix, 22) == 1
                && MagicHelpers.Le16(prefix, 24) is ushort bpp && bpp is 1 or 4 or 8 or 24
                ? "bmp" : null;
        }
        if (prefix.Length < 54 || offBits.Value < 54) return null;
        if (dib.Value is not (40 or 52 or 56 or 108 or 124)) return null;
        uint? width = MagicHelpers.Le32(prefix, 18);
        uint? height = MagicHelpers.Le32(prefix, 22);
        if (width is null || height is null || width.Value < 1 || width.Value > 0x7FFFFFFF || height.Value < 1)
            return null;
        return "bmp";
    }


    private static string? Bpg(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x42, 0x50, 0x47, 0xFB }))) return null;
        return "bpg";
    }

private static string? Bzip(ReadOnlySpan<byte> prefix)
{
    // BZh[1-9] + 0x314159265359 (blocksize + magic)
    if (prefix.Length < 10) return null;
    if (prefix[0] != (byte)'B' || prefix[1] != (byte)'Z' || prefix[2] != (byte)'h') return null;
    if (prefix[3] < (byte)'1' || prefix[3] > (byte)'9') return null;
    return "bzip";
}

    private static string? Cab(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (cab): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Cram(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("CRAM"u8))) return null;
        return "cram";
    }

    private static string? Crx(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 16) return null;
        if (!(prefix.StartsWith("Cr24"u8))) return null;
        return "crx";
    }

    private static string? Cubin(ReadOnlySpan<byte> prefix)
    {
        // ELF magic + e_machine == EM_CUDA (190), endianness from EI_DATA.
        if (prefix.Length < 20) return null;
        if (!prefix.StartsWith(new byte[] { 0x7F, 0x45, 0x4C, 0x46 })) return null;
        byte eiData = prefix[5];
        ushort machine = eiData switch
        {
            1 => MagicHelpers.Le16(prefix, 18) ?? 0,
            2 => MagicHelpers.Be16(prefix, 18) ?? 0,
            _ => (ushort)0,
        };
        return machine == 190 ? "cubin" : null;
    }

    private static string? Dbase(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
    {
        // Complex verifier (dbase): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Degas(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
    {
        // Complex verifier (degas): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Dex(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x64, 0x65, 0x78, 0x0A }))) return null;
        return "dex";
    }

private static string? Dicom(ReadOnlySpan<byte> prefix)
{
    if (prefix.Length < 132) return null;
    return MagicHelpers.EqAt(prefix, 128, "DICM"u8) ? "dicom" : null;
}

    private static string? Dsstore(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (dsstore): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Duckdb(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4096) return null;
        if (!MagicHelpers.EqAt(prefix, 8, "DUCK"u8)) return null;
        if (MagicHelpers.Le32(prefix, 12) is not uint lo || lo < 1) return null;
        if (MagicHelpers.Le32(prefix, 16) != 0) return null;
        return "duckdb";
    }


    private static string? Dwg(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("AC10"u8))) return null;
        return "dwg";
    }

    private static string? Emf(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (emf): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

private static string? EpubPrefix(ReadOnlySpan<byte> prefix) => ZipMagic.EpubPrefix(prefix);

    private static string? Ese(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (ese): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Fbx(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x4B, 0x61, 0x79, 0x64, 0x61, 0x72, 0x61, 0x20, 0x46, 0x42, 0x58, 0x20, 0x42, 0x69, 0x6E, 0x61, 0x72, 0x79, 0x20, 0x20, 0x00, 0x1A, 0x00 }) || prefix.StartsWith("; FBX "u8))) return null;
        return "fbx";
    }

    private static string? Fits(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (fits): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Flac(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("fLaC"u8))) return null;
        return "flac";
    }

    private static string? Flv(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x46, 0x4C, 0x56, 0x01 }))) return null;
        return "flv";
    }

private static string? Ftyp(ReadOnlySpan<byte> prefix)
{
    // ISO-BMFF: size(4) + "ftyp" at 4; brand decides heif/avif/mp4/...
    if (prefix.Length < 12) return null;
    if (!MagicHelpers.EqAt(prefix, 4, "ftyp"u8)) return null;
    ReadOnlySpan<byte> brand = prefix.Slice(8, 4);
    // HEIF family
    if (brand.SequenceEqual("heic"u8) || brand.SequenceEqual("heix"u8) || brand.SequenceEqual("hevc"u8)
        || brand.SequenceEqual("hevx"u8) || brand.SequenceEqual("mif1"u8) || brand.SequenceEqual("msf1"u8)
        || brand.SequenceEqual("heim"u8) || brand.SequenceEqual("heis"u8) || brand.SequenceEqual("hevm"u8)
        || brand.SequenceEqual("hevs"u8))
        return "heif";
    if (brand.SequenceEqual("avif"u8) || brand.SequenceEqual("avis"u8))
        return "avif";
    // Many other brands map to mp4 in model; resolvers may return null for non-veto brands.
    // Upstream ftyp() returns specific labels; under-match is safer than wrong label.
    if (brand.SequenceEqual("isom"u8) || brand.SequenceEqual("iso2"u8) || brand.SequenceEqual("mp41"u8)
        || brand.SequenceEqual("mp42"u8) || brand.SequenceEqual("M4V "u8) || brand.SequenceEqual("M4A "u8))
        return null; // model handles mp4
    return null;
}

    private static string? GeoPackage(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 100) return null;
        if (!prefix.StartsWith("SQLite format 3\0"u8)) return null;
        uint? id = MagicHelpers.Be32(prefix, 68);
        if (id is null) return null;
        uint gpkg = MagicHelpers.TagU32("GPKG"u8);
        uint gp10 = MagicHelpers.TagU32("GP10"u8);
        uint gp11 = MagicHelpers.TagU32("GP11"u8);
        return id == gpkg || id == gp10 || id == gp11 ? "geopackage" : null;
    }


    private static string? Gguf(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("GGUF"u8))) return null;
        return "gguf";
    }

    private static string? Gif(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("GIF87a"u8) || prefix.StartsWith("GIF89a"u8))) return null;
        return "gif";
    }

    private static string? Grib(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("GRIB"u8))) return null;
        return "grib";
    }

private static string? Gzip(ReadOnlySpan<byte> prefix)
{
    if (prefix.Length < 20 || !prefix.StartsWith(new byte[] { 0x1F, 0x8B, 0x08 })) return null;
    if (prefix[3] > 0x1F) return null;
    return "gzip";
}

    private static string? Hdf4(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (hdf4): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Hlp(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (hlp): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Hve(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("regf"u8))) return null;
        return "hve";
    }

    private static string? Icc(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 132) return null;
        if (!MagicHelpers.EqAt(prefix, 36, "acsp"u8)) return null;
        ReadOnlySpan<byte> dev = prefix.Slice(12, 4);
        bool ok = dev.SequenceEqual(new byte[] { 0, 0, 0, 0 })
            || dev.SequenceEqual("scnr"u8) || dev.SequenceEqual("mntr"u8)
            || dev.SequenceEqual("prtr"u8) || dev.SequenceEqual("link"u8)
            || dev.SequenceEqual("abst"u8) || dev.SequenceEqual("spac"u8)
            || dev.SequenceEqual("nmcl"u8) || dev.SequenceEqual("cenc"u8)
            || dev.SequenceEqual("mid "u8) || dev.SequenceEqual("mlnk"u8)
            || dev.SequenceEqual("mvis"u8);
        return ok ? "icc" : null;
    }


    private static string? Icns(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("icns"u8))) return null;
        return "icns";
    }

    private static string? Ico(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (ico): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Ilbm(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 12 || !prefix.StartsWith("FORM"u8)) return null;
        return MagicHelpers.EqAt(prefix, 8, "ILBM"u8)
            || MagicHelpers.EqAt(prefix, 8, "ACBM"u8)
            || MagicHelpers.EqAt(prefix, 8, "PBM "u8)
            ? "ilbm" : null;
    }


    private static string? IntelHex(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (intelhex): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Jng(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (jng): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Jp2OrJxl(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (jp2_or_jxl): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Koala(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (koala): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? LlvmBitcode(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
    {
        if (prefix.Length < 20) return null;
        if (!(prefix.StartsWith(new byte[] { 0x42, 0x43, 0xC0, 0xDE }))) return null;
        return "llvm_bitcode";
    }

    private static string? Lnk(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (lnk): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Lrz(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("LRZI"u8))) return null;
        return "lrz";
    }

    private static string? LuaBytecode(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 18) return null;
        if (!(prefix.StartsWith(new byte[] { 0x1B, 0x4C, 0x75, 0x61 }) || prefix.StartsWith(new byte[] { 0x1B, 0x4C, 0x4A }))) return null;
        return "luabytecode";
    }

    private static string? Lz4(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
    {
        // Complex verifier (lz4): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Lzip(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("LZIP"u8))) return null;
        return "lz";
    }

    private static string? Macho(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (macho): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Mat(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("MATLAB 5"u8))) return null;
        return "mat";
    }

    private static string? Midi(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (midi): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? MiniDump(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (minidump): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Mkv(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("matroska"u8))) return null;
        return "mkv";
    }

    private static string? Mpegts(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (mpegts): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? MsCompress(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x53, 0x5A, 0x44, 0x44, 0x88, 0xF0, 0x27, 0x33 }))) return null;
        return "mscompress";
    }

    private static string? NetCdf(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("CDF"u8))) return null;
        return "netcdf";
    }

    private static string? Nrrd(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("NRRD000"u8))) return null;
        return "nrrd";
    }

    private static string? Ogg(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x4F, 0x67, 0x67, 0x53, 0x00 }))) return null;
        return "ogg";
    }

    private static string? One(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (one): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? OsmPbf(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x0A, 0x09, 0x4F, 0x53, 0x4D, 0x48, 0x65, 0x61, 0x64, 0x65, 0x72 }))) return null;
        return "osm";
    }

    private static string? Palmos(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (palmos): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Parquet(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("PAR1"u8))) return null;
        return "parquet";
    }

    private static string? Pbm(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (pbm): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Pcap(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (pcap): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? PcapNg(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x0A, 0x0D, 0x0D, 0x0A }))) return null;
        return "pcapng";
    }

    private static string? Pdb(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x4D, 0x69, 0x63, 0x72, 0x6F, 0x73, 0x6F, 0x66, 0x74, 0x20, 0x43, 0x2F, 0x43, 0x2B, 0x2B, 0x20, 0x4D, 0x53, 0x46, 0x20, 0x37, 0x2E, 0x30, 0x30, 0x0D, 0x0A, 0x1A, 0x44, 0x53, 0x00, 0x00, 0x00 }) || prefix.StartsWith(new byte[] { 0x4D, 0x69, 0x63, 0x72, 0x6F, 0x73, 0x6F, 0x66, 0x74, 0x20, 0x43, 0x2F, 0x43, 0x2B, 0x2B, 0x20, 0x70, 0x72, 0x6F, 0x67, 0x72, 0x61, 0x6D, 0x20, 0x64, 0x61, 0x74, 0x61, 0x62, 0x61, 0x73, 0x65, 0x20, 0x32, 0x2E, 0x30, 0x30, 0x0D, 0x0A, 0x1A, 0x4A, 0x47, 0x00, 0x00 }))) return null;
        return "pdb";
    }

private static string? Pe(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
{
    PeFacts.Facts facts = PeFacts.Analyze(prefix, ctx.Size);
    return PeFacts.IsWindowsPebin(in facts) ? "pebin" : null;
}

    private static string? PemOrPgp(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("-----BEGIN PGP "u8))) return null;
        return "pgp";
    }

    private static string? Ply(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (ply): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? PostgresDump(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 11) return null;
        if (!(prefix.StartsWith(new byte[] { 0x50, 0x47, 0x44, 0x4D, 0x50, 0x01 }))) return null;
        return "postgres_dump";
    }

    private static string? PostScript(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("%!PS-Adobe"u8))) return null;
        return "postscript";
    }

    private static string? Psd(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("8BPS"u8))) return null;
        return "psd";
    }

    private static string? PythonBytecode(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (pythonbytecode): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Qoi(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("qoif"u8))) return null;
        return "qoi";
    }

private static string? Rar(ReadOnlySpan<byte> prefix)
{
    if (prefix.Length < 8) return null;
    if (prefix.StartsWith(new byte[] { 0x52, 0x61, 0x72, 0x21, 0x1A, 0x07, 0x00 })
        || prefix.StartsWith(new byte[] { 0x52, 0x61, 0x72, 0x21, 0x1A, 0x07, 0x01, 0x00 })
        || prefix.StartsWith("RE~^"u8))
        return "rar";
    return null;
}

    private static string? RedisRdb(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("REDIS"u8))) return null;
        return "redis_rdb";
    }

    private static string? Riff(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 12) return null;
        if (prefix.StartsWith("RIFF"u8))
        {
            if (prefix.Length >= 24
                && MagicHelpers.EqAt(prefix, 8, "AVI LIST"u8)
                && MagicHelpers.EqAt(prefix, 20, "hdrl"u8)
                && MagicHelpers.Le32(prefix, 4) is uint cs && cs >= 16
                && MagicHelpers.Le32(prefix, 16) is uint ls && ls >= 4)
                return "avi";
            if (MagicHelpers.EqAt(prefix, 8, "WAVE"u8) && MagicHelpers.Le32(prefix, 4) is uint w && w >= 4)
                return "wav";
            if (MagicHelpers.EqAt(prefix, 8, "ACON"u8))
                return "ani";
            return null;
        }
        if (prefix.StartsWith("RIFX"u8))
        {
            if (MagicHelpers.EqAt(prefix, 8, "WAVE"u8) && MagicHelpers.Be32(prefix, 4) is uint w && w >= 4)
                return "wav";
            return null;
        }
        if (prefix.Length >= 48
            && prefix.StartsWith("RF64"u8)
            && MagicHelpers.Le32(prefix, 4) == 0xFFFFFFFF
            && MagicHelpers.EqAt(prefix, 8, "WAVEds64"u8))
            return "wav";
        return null;
    }


    private static string? Rpm(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (rpm): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Rzip(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x52, 0x5A, 0x49, 0x50, 0x02 }))) return null;
        return "rzip";
    }

    private static string? Sas(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("SAS FILE"u8))) return null;
        return "sas";
    }

private static string? SevenZip(ReadOnlySpan<byte> prefix)
{
    // "7z\xbc\xaf\x27\x1c\x00"
    if (prefix.Length < 32 || !prefix.StartsWith(new byte[] { 0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C, 0x00 })) return null;
    return "sevenzip";
}

    private static string? ShapeFile(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (shapefile): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? SketchUp(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x0E, 0x53, 0x6B, 0x65, 0x74, 0x63, 0x68, 0x55, 0x70, 0x20, 0x4D, 0x6F, 0x64, 0x65, 0x6C, 0x08 }))) return null;
        return "sketchup";
    }

    private static string? Spirv(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
    {
        // Complex verifier (spirv): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Spss(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("$FL2"u8) || prefix.StartsWith("$FL3"u8))) return null;
        return "spss";
    }

    private static string? Stata(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("<stata_dta><header><release>"u8))) return null;
        return "stata";
    }

    private static string? Step(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("ISO-10303-21;"u8))) return null;
        return "step";
    }

    private static string? Swf(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 17) return null;
        if (!(prefix.StartsWith("FWS"u8) || prefix.StartsWith("CWS"u8) || prefix.StartsWith("ZWS"u8))) return null;
        return "swf";
    }

    private static string? ThreeDsm(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 16 || !prefix.StartsWith("MM"u8)) return null;
        uint? rootLen = MagicHelpers.Le32(prefix, 2);
        if (rootLen is null || rootLen.Value < 16) return null;
        bool version = MagicHelpers.EqAt(prefix, 6, new byte[] { 0x02, 0x00, 0x0A, 0x00, 0x00, 0x00 })
            && MagicHelpers.Le32(prefix, 12) is uint v && v is 3 or 4;
        bool editor = MagicHelpers.EqAt(prefix, 6, "=="u8)
            && MagicHelpers.Le32(prefix, 8) is uint el && el >= 6;
        return version || editor ? "3dsm" : null;
    }


    private static string? ThreeDsx(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("3DSX"u8))) return null;
        return "3dsx";
    }

    private static string? Torrent(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("d13:announce-list"u8) || prefix.StartsWith("d4:info"u8) || prefix.StartsWith("d8:announce"u8) || prefix.StartsWith("d7:comment"u8) || prefix.StartsWith("d8:announce"u8) || prefix.StartsWith("d13:announce-listll"u8))) return null;
        return "torrent";
    }

    private static string? Uf2(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (uf2): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? UnixCompress(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (unixcompress): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Vhd(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("conectix"u8))) return null;
        return "vhd";
    }

    private static string? Vib(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith(new byte[] { 0x21, 0x3C, 0x61, 0x72, 0x63, 0x68, 0x3E, 0x0A }))) return null;
        return "vib";
    }

    private static string? Wad(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("IWAD"u8) || prefix.StartsWith("PWAD"u8))) return null;
        return "wad";
    }

    private static string? Wasm(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (wasm): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Woff(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("wOFF"u8))) return null;
        return "woff";
    }

    private static string? Woff2(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("wOF2"u8))) return null;
        return "woff2";
    }

    private static string? Xar(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("xar!"u8))) return null;
        return "xar";
    }

    private static string? Xcf(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 4) return null;
        if (!(prefix.StartsWith("gimp xcf "u8))) return null;
        return "xcf";
    }

    private static string? Xcoff(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (xcoff): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

private static string? Zip(ReadOnlySpan<byte> prefix, in MagicCtx ctx)
{
    ReadOnlySpan<byte> tail = ctx.HasTail
        ? ctx.Tail
        : (ctx.Size == (ulong)prefix.Length ? prefix : ReadOnlySpan<byte>.Empty);
    return ZipMagic.Identify(prefix, ctx.Size, tail);
}

    private static string? Zst(ReadOnlySpan<byte> prefix)
    {
        // Complex verifier (zst): structural checks not yet fully auto-ported; avoid false positives.
        return null;
    }

    private static string? Glb(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 24) return null;
        if (!prefix.StartsWith("glTF"u8)) return null;
        if (MagicHelpers.Le32(prefix, 4) != 2) return null;
        uint? total = MagicHelpers.Le32(prefix, 8);
        uint? chunk = MagicHelpers.Le32(prefix, 12);
        if (total is null || chunk is null) return null;
        if (total.Value < 24 || total.Value % 4 != 0) return null;
        if (chunk.Value < 4 || chunk.Value % 4 != 0) return null;
        if (!MagicHelpers.EqAt(prefix, 16, "JSON"u8)) return null;
        int i = 20;
        int limit = Math.Min(prefix.Length, 20 + 65);
        while (i < limit)
        {
            byte b = prefix[i];
            if (b is not ((byte)' ' or (byte)'\t' or (byte)'\r' or (byte)'\n')) break;
            i++;
        }
        if (i - 20 > 64 || i >= prefix.Length || prefix[i] != (byte)'{') return null;
        return "gltf";
    }

    private static string? Npy(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 10) return null;
        if (!prefix.StartsWith(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' })) return null;
        return "npy";
    }

    private static string? GltfJson(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 32) return null;
        int i = 0;
        if (prefix.Length >= 3 && prefix[0] == 0xEF && prefix[1] == 0xBB && prefix[2] == 0xBF) i = 3;
        while (i < prefix.Length && prefix[i] is ((byte)' ' or (byte)'\t' or (byte)'\r' or (byte)'\n')) i++;
        if (i >= prefix.Length || prefix[i] != (byte)'{') return null;
        if (prefix.IndexOf("\"asset\""u8) < 0) return null;
        if (prefix.IndexOf("\"version\""u8) < 0) return null;
        return "gltf";
    }

    private static string? Safetensors(ReadOnlySpan<byte> prefix)
    {
        if (prefix.Length < 9) return null;
        if (MagicHelpers.Le32(prefix, 4) != 0) return null;
        uint? lo = MagicHelpers.Le32(prefix, 0);
        if (lo is null || lo.Value < 2) return null;
        if (prefix[8] != (byte)'{') return null;
        return "safetensors";
    }

}
