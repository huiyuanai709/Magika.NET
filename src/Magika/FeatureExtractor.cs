using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Magika;

/// <summary>
/// Beg/end feature extraction matching the Magika Rust library for models with no middle window.
/// </summary>
internal static class FeatureExtractor
{
    internal static int[] Extract(ReadOnlySpan<byte> content, ModelDefinition model)
    {
        int block = Math.Min(model.BlockSize, content.Length);
        ReadOnlySpan<byte> prefix = content[..block];
        ReadOnlySpan<byte> suffix = content[^block..];
        return ExtractBlocks(prefix, suffix, model);
    }

    internal static int[] ExtractBlocks(ReadOnlySpan<byte> prefix, ReadOnlySpan<byte> suffix, ModelDefinition model)
    {
        int[] features = new int[model.FeaturesSize];
        ExtractBlocks(prefix, suffix, model, features);
        return features;
    }

    internal static void ExtractBlocks(ReadOnlySpan<byte> prefix, ReadOnlySpan<byte> suffix, ModelDefinition model, Span<ushort> features)
    {
        if (features.Length != model.FeaturesSize)
        {
            throw new ArgumentException("Feature buffer length does not match the model.", nameof(features));
        }

        ReadOnlySpan<byte> beg = StripStart(prefix);
        ReadOnlySpan<byte> end = StripEnd(suffix);
        int begCopy = Math.Min(model.BegSize, beg.Length);
        int endCopy = Math.Min(model.EndSize, end.Length);
        int dest = model.BegSize + model.EndSize - endCopy;
        if (begCopy < dest)
        {
            features.Slice(begCopy, dest - begCopy).Fill((ushort)model.PaddingToken);
        }

        Widen(beg[..begCopy], features[..begCopy]);
        Widen(end.Slice(end.Length - endCopy, endCopy), features.Slice(dest, endCopy));
    }

    internal static void ExtractBlocks(ReadOnlySpan<byte> prefix, ReadOnlySpan<byte> suffix, ModelDefinition model, Span<int> features)
    {
        if (features.Length != model.FeaturesSize)
        {
            throw new ArgumentException("Feature buffer length does not match the model.", nameof(features));
        }

        ReadOnlySpan<byte> beg = StripStart(prefix);
        ReadOnlySpan<byte> end = StripEnd(suffix);
        int begCopy = Math.Min(model.BegSize, beg.Length);
        int endCopy = Math.Min(model.EndSize, end.Length);
        int dest = model.BegSize + model.EndSize - endCopy;
        if (begCopy < dest)
        {
            features.Slice(begCopy, dest - begCopy).Fill(model.PaddingToken);
        }

        Widen(beg[..begCopy], features[..begCopy]);
        Widen(end.Slice(end.Length - endCopy, endCopy), features.Slice(dest, endCopy));
    }

    /// <summary>
    /// True when the stripped prefix can fill <see cref="ModelDefinition.MinFileSizeForDl"/>.
    /// The end block is written at the tail of the feature vector, so it cannot change an index below <see cref="ModelDefinition.BegSize"/>.
    /// Returns true when the gate sits in the end block, which this model does not do.
    /// </summary>
    internal static bool HasModelPrefix(ReadOnlySpan<byte> prefix, ModelDefinition model)
    {
        int needed = model.MinFileSizeForDl;
        if ((uint)(needed - 1) >= (uint)model.BegSize)
        {
            return true;
        }

        int i = 0;
        while (i < prefix.Length && IsWhitespace(prefix[i]))
        {
            i++;
        }

        return prefix.Length - i >= needed;
    }

    internal static bool HasEnoughContent(ReadOnlySpan<int> features, ModelDefinition model) =>
        (uint)(model.MinFileSizeForDl - 1) < (uint)features.Length
        && features[model.MinFileSizeForDl - 1] != model.PaddingToken;

    internal static bool HasEnoughContent(ReadOnlySpan<ushort> features, ModelDefinition model) =>
        (uint)(model.MinFileSizeForDl - 1) < (uint)features.Length
        && features[model.MinFileSizeForDl - 1] != (ushort)model.PaddingToken;

    internal static string ClassifySmall(ReadOnlySpan<byte> prefix, long fileLength)
    {
        Utf8Kind kind = InspectUtf8(prefix);
        if (kind == Utf8Kind.Valid)
        {
            return "txt";
        }

        if (kind == Utf8Kind.IncompleteTail && fileLength != prefix.Length)
        {
            return "txt";
        }

        return "unknown";
    }

    internal static Utf8Kind InspectUtf8(ReadOnlySpan<byte> data)
    {
        int i = 0;
        while (i < data.Length)
        {
            byte lead = data[i];
            if (lead <= 0x7F)
            {
                i++;
                continue;
            }

            if (!TrySequenceLength(lead, out int need, out byte min2, out byte max2))
            {
                return Utf8Kind.Invalid;
            }

            int available = data.Length - i;
            if (available < need)
            {
                return ValidPrefix(data[i..]) ? Utf8Kind.IncompleteTail : Utf8Kind.Invalid;
            }

            if (!IsCont(data[i + 1], min2, max2))
            {
                return Utf8Kind.Invalid;
            }

            for (int c = 2; c < need; c++)
            {
                if (!IsCont(data[i + c], 0x80, 0xBF))
                {
                    return Utf8Kind.Invalid;
                }
            }

            i += need;
        }

        return Utf8Kind.Valid;
    }

    private static bool ValidPrefix(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0 || !TrySequenceLength(data[0], out int need, out byte min2, out byte max2))
        {
            return false;
        }

        if (data.Length >= need)
        {
            return false;
        }

        if (data.Length >= 2 && !IsCont(data[1], min2, max2))
        {
            return false;
        }

        for (int c = 2; c < data.Length; c++)
        {
            if (!IsCont(data[c], 0x80, 0xBF))
            {
                return false;
            }
        }

        return true;
    }

    private static bool TrySequenceLength(byte lead, out int need, out byte min2, out byte max2)
    {
        min2 = 0x80;
        max2 = 0xBF;
        if (lead >= 0xC2 && lead <= 0xDF)
        {
            need = 2;
            return true;
        }

        if (lead == 0xE0)
        {
            need = 3;
            min2 = 0xA0;
            return true;
        }

        if (lead >= 0xE1 && lead <= 0xEC)
        {
            need = 3;
            return true;
        }

        if (lead == 0xED)
        {
            need = 3;
            max2 = 0x9F;
            return true;
        }

        if (lead >= 0xEE && lead <= 0xEF)
        {
            need = 3;
            return true;
        }

        if (lead == 0xF0)
        {
            need = 4;
            min2 = 0x90;
            return true;
        }

        if (lead >= 0xF1 && lead <= 0xF3)
        {
            need = 4;
            return true;
        }

        if (lead == 0xF4)
        {
            need = 4;
            max2 = 0x8F;
            return true;
        }

        need = 0;
        return false;
    }

    private static bool IsCont(byte value, byte min, byte max) => value >= min && value <= max;

    private static ReadOnlySpan<byte> StripStart(ReadOnlySpan<byte> data)
    {
        int i = 0;
        while (i < data.Length && IsWhitespace(data[i]))
        {
            i++;
        }

        return data[i..];
    }

    private static ReadOnlySpan<byte> StripEnd(ReadOnlySpan<byte> data)
    {
        int i = data.Length;
        while (i > 0 && IsWhitespace(data[i - 1]))
        {
            i--;
        }

        return data[..i];
    }

    private static bool IsWhitespace(byte value) =>
        value is (byte)' ' or (byte)'\t' or (byte)'\n' or (byte)'\r' or 0x0C or 0x0B;

    private static void Widen(ReadOnlySpan<byte> source, Span<ushort> destination)
    {
        ref byte src = ref MemoryMarshal.GetReference(source);
        ref ushort dst = ref MemoryMarshal.GetReference(destination);
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; i + 32 <= source.Length; i += 32)
            {
                (Vector256<ushort> lower, Vector256<ushort> upper) = Vector256.Widen(Vector256.LoadUnsafe(ref src, (nuint)i));
                lower.StoreUnsafe(ref dst, (nuint)i);
                upper.StoreUnsafe(ref dst, (nuint)(i + 16));
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (; i + 16 <= source.Length; i += 16)
            {
                (Vector128<ushort> lower, Vector128<ushort> upper) = Vector128.Widen(Vector128.LoadUnsafe(ref src, (nuint)i));
                lower.StoreUnsafe(ref dst, (nuint)i);
                upper.StoreUnsafe(ref dst, (nuint)(i + 8));
            }
        }

        for (; i < source.Length; i++)
        {
            destination[i] = source[i];
        }
    }

    private static void Widen(ReadOnlySpan<byte> source, Span<int> destination)
    {
        ref byte src = ref MemoryMarshal.GetReference(source);
        ref int dst = ref MemoryMarshal.GetReference(destination);
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; i + 32 <= source.Length; i += 32)
            {
                (Vector256<ushort> lower, Vector256<ushort> upper) = Vector256.Widen(Vector256.LoadUnsafe(ref src, (nuint)i));
                (Vector256<uint> a, Vector256<uint> b) = Vector256.Widen(lower);
                (Vector256<uint> c, Vector256<uint> d) = Vector256.Widen(upper);
                a.As<uint, int>().StoreUnsafe(ref dst, (nuint)i);
                b.As<uint, int>().StoreUnsafe(ref dst, (nuint)(i + 8));
                c.As<uint, int>().StoreUnsafe(ref dst, (nuint)(i + 16));
                d.As<uint, int>().StoreUnsafe(ref dst, (nuint)(i + 24));
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (; i + 16 <= source.Length; i += 16)
            {
                (Vector128<ushort> lower, Vector128<ushort> upper) = Vector128.Widen(Vector128.LoadUnsafe(ref src, (nuint)i));
                (Vector128<uint> a, Vector128<uint> b) = Vector128.Widen(lower);
                (Vector128<uint> c, Vector128<uint> d) = Vector128.Widen(upper);
                a.As<uint, int>().StoreUnsafe(ref dst, (nuint)i);
                b.As<uint, int>().StoreUnsafe(ref dst, (nuint)(i + 4));
                c.As<uint, int>().StoreUnsafe(ref dst, (nuint)(i + 8));
                d.As<uint, int>().StoreUnsafe(ref dst, (nuint)(i + 12));
            }
        }

        for (; i < source.Length; i++)
        {
            destination[i] = source[i];
        }
    }

    internal enum Utf8Kind
    {
        Valid,
        IncompleteTail,
        Invalid,
    }
}
