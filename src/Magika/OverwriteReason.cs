namespace Magika;

/// <summary>Why the reported label differs from the model's raw argmax.</summary>
public enum OverwriteReason
{
    /// <summary>The score was below the active confidence threshold.</summary>
    LowConfidence,

    /// <summary>The model label is rewritten by the model's overwrite map.</summary>
    OverwriteMap,

    /// <summary>
    /// Reserved for upstream format-rule vetoes. This build does not ship the bundled rule engine,
    /// so identification does not produce this reason.
    /// </summary>
    RulesVeto,
}
