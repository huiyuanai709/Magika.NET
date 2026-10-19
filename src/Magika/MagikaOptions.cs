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

    /// <summary>
    /// Feature rows per forward pass for <see cref="MagikaSession.IdentifyPaths"/>,
    /// <see cref="MagikaSession.IdentifyPathsAsync"/>, and <see cref="MagikaSession.IdentifyFeaturesBatch"/>.
    /// The default is 8. On a 4-core machine, batches of 2 to 8 were faster per file than batch 1.
    /// Larger batches keep more activation memory live. Must be from 1 to 64.
    /// </summary>
    public int BatchSize { get; init; } = 8;

    /// <summary>
    /// How many inference lanes multi-file identification may run at once.
    /// <c>0</c> (the default) picks the count.
    /// For the bundled specialized model that count is one lane, and the convolution uses every processor.
    /// Several lanes each running that parallel product oversubscribe the machine and the cache.
    /// A graph that stays on the interpreter still uses one lane per processor.
    /// Each lane shares the model weights and keeps its own activation scratch.
    /// Lanes are created on the first multi-file call. When there is more than one lane, the convolution's thread count is split across them so the product stays near the processor count.
    /// Single-file identification always uses one lane and does not create the pool.
    /// Negative values are rejected. An explicit count is used as given, with no cap.
    /// </summary>
    public int InferenceWorkers { get; init; }
}
