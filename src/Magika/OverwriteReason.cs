namespace Magika;

/// <summary>Why the reported label differs from the model's raw argmax.</summary>
public enum OverwriteReason
{
    /// <summary>The score was below the active confidence threshold.</summary>
    LowConfidence,

    /// <summary>The model label is rewritten by the model's overwrite map.</summary>
    OverwriteMap,

    /// <summary>
    /// A format resolver overwrote or supplied the label.
    /// Magika.NET currently ships the upstream ZIP→MSIX central-directory check
    /// (<c>AppxManifest.xml</c> + <c>AppxBlockMap.xml</c>); other rule-only labels stay catalogued only.
    /// </summary>
    RulesVeto,
}
