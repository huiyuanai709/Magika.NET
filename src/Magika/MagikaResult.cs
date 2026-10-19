namespace Magika;

/// <summary>Content-type identification of one file or buffer.</summary>
public sealed class MagikaResult
{
    internal MagikaResult(
        string? path,
        ContentTypeInfo contentType,
        ContentTypeInfo? modelContentType,
        float score,
        OverwriteReason? overwriteReason)
    {
        Path = path;
        ContentType = contentType;
        ModelContentType = modelContentType;
        Score = score;
        OverwriteReason = overwriteReason;
    }

    /// <summary>Path that was identified, when identification started from a path.</summary>
    public string? Path { get; }

    /// <summary>Final content type after thresholds and the overwrite map.</summary>
    public ContentTypeInfo ContentType { get; }

    /// <summary>
    /// Raw model argmax, before the confidence fallback and overwrite map.
    /// <see langword="null"/> when the model did not run.
    /// </summary>
    public ContentTypeInfo? ModelContentType { get; }

    /// <summary>
    /// Model probability of <see cref="ModelContentType"/>, or 1 when a heuristic decided the type.
    /// </summary>
    public float Score { get; }

    /// <summary>
    /// Why <see cref="ContentType"/> differs from <see cref="ModelContentType"/>.
    /// <see langword="null"/> when they are the same, or when the model did not run.
    /// </summary>
    public OverwriteReason? OverwriteReason { get; }

    public override string ToString() =>
        ModelContentType is null
            ? ContentType.Label
            : $"{ContentType.Label} ({Score:0.000})";
}
