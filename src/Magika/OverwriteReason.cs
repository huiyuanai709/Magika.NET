namespace Magika;

/// <summary>Why the reported label differs from the model's raw argmax.</summary>
public enum OverwriteReason
{
    /// <summary>The score was below the active confidence threshold.</summary>
    LowConfidence,

    /// <summary>The model label is rewritten by the model's overwrite map.</summary>
    OverwriteMap,

    /// <summary>
    /// A format resolver supplied the label (magic / ZIP subtype / PE) before the model ran.
    /// </summary>
    RulesVeto,

    /// <summary>
    /// The model predicted a label in upstream <c>resolvers::VETOS</c> but no resolver matched,
    /// so the result fell back to <c>txt</c> or <c>unknown</c>.
    /// </summary>
    ResolversVeto,
}
