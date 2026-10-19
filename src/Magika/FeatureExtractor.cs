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
        ReadOnlySpan<byte> beg = StripStart(prefix);
        ReadOnlySpan<byte> end = StripEnd(suffix);
        int[] features = new int[model.FeaturesSize];
        Array.Fill(features, model.PaddingToken);

        int begCopy = Math.Min(model.BegSize, beg.Length);
        for (int i = 0; i < begCopy; i++)
        {
            features[i] = beg[i];
        }

        int endCopy = Math.Min(model.EndSize, end.Length);
        int dest = model.BegSize + model.EndSize - endCopy;
        int src = end.Length - endCopy;
        for (int i = 0; i < endCopy; i++)
        {
            features[dest + i] = end[src + i];
        }

        return features;
    }

    internal static bool HasEnoughContent(int[] features, ModelDefinition model) =>
        features[model.MinFileSizeForDl - 1] != model.PaddingToken;

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

    internal enum Utf8Kind
    {
        Valid,
        IncompleteTail,
        Invalid,
    }
}
