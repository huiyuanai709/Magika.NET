namespace Magika;

/// <summary>Options for a <see cref="MagikaSession"/>.</summary>
public sealed class MagikaOptions
{
    public static MagikaOptions Default { get; } = new();

    /// <summary>Minimum confidence required to keep a model label. Defaults to high confidence.</summary>
    public PredictionMode PredictionMode { get; init; } = PredictionMode.HighConfidence;

    /// <summary>When false, a symlink is reported as <c>symlink</c> instead of the target.</summary>
    public bool FollowSymlinks { get; init; } = true;

    /// <summary>When false, skip the model and use only the empty-file and UTF-8 size heuristic.</summary>
    public bool UseModel { get; init; } = true;
}
