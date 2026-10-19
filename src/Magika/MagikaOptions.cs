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
    /// <c>0</c> (the default) uses one lane per processor, capped at 8.
    /// Each lane is a managed ONNX session that shares the model weights and keeps its own activation scratch.
    /// Lanes are created on the first multi-file call. The convolution product's thread count is split across lanes.
    /// Single-file identification always uses one lane and does not create the pool.
    /// Must be from 0 to 32.
    /// </summary>
    public int InferenceWorkers { get; init; }
}
