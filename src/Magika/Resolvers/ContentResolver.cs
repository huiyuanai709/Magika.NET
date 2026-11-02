using System.Runtime.CompilerServices;

namespace Magika.Resolvers;

/// <summary>
/// Entry point for google/magika content resolvers: magic cascade (incl. ZIP subtypes + PE),
/// then optional model with <see cref="ResolverVetos"/>.
/// </summary>
internal static class ContentResolver
{
    internal const int PrefixLimit = ZipFacts.PrefixLimit;
    internal const int TailBytes = ZipFacts.TailBytes;
    internal const int DirectoryMaxBytes = ZipFacts.DirectoryMaxBytes;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool WantsZipTail(ReadOnlySpan<byte> prefix) => ZipFacts.WantsTail(prefix);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static long? DirectoryStart(ReadOnlySpan<byte> tail, long size) =>
        ZipFacts.DirectoryStart(tail, size);

    /// <summary>
    /// Runs the upstream magic cascade. <paramref name="tail"/> may be empty when the prefix
    /// does not need a ZIP tail (or the caller could not obtain one).
    /// </summary>
    internal static string? Identify(ReadOnlySpan<byte> prefix, long size, ReadOnlySpan<byte> tail)
    {
        if (size <= 0 || prefix.IsEmpty)
        {
            return null;
        }

        // Upstream uses up to PREFIX_LIMIT (4096) of the first block.
        if (prefix.Length > PrefixLimit)
        {
            prefix = prefix[..PrefixLimit];
        }

        var ctx = new MagicCtx
        {
            Prefix = prefix,
            Size = (ulong)size,
            Tail = tail,
            HasTail = !tail.IsEmpty,
        };
        return Magic.Identify(in ctx);
    }
}
