namespace Magika;

/// <summary>Metadata for one Magika content type.</summary>
public sealed class ContentTypeInfo
{
    public ContentTypeInfo(
        string label,
        string mimeType,
        string group,
        string description,
        string[] extensions,
        bool isText,
        bool modelSupport,
        bool rulesSupport)
    {
        Label = label;
        MimeType = mimeType;
        Group = group;
        Description = description;
        Extensions = extensions;
        IsText = isText;
        ModelSupport = modelSupport;
        RulesSupport = rulesSupport;
    }

    /// <summary>Stable Magika label, such as <c>pdf</c> or <c>elf</c>.</summary>
    public string Label { get; }

    /// <summary>MIME type associated with the label.</summary>
    public string MimeType { get; }

    /// <summary>High-level group such as <c>document</c>, <c>image</c>, or <c>code</c>.</summary>
    public string Group { get; }

    /// <summary>Short human-readable description.</summary>
    public string Description { get; }

    /// <summary>Common file extensions, without the leading dot.</summary>
    public IReadOnlyList<string> Extensions { get; }

    /// <summary>Whether Magika treats the type as text.</summary>
    public bool IsText { get; }

    /// <summary>Whether the bundled model can emit this label before overwrite.</summary>
    public bool ModelSupport { get; }

    /// <summary>Whether upstream's format rules can emit this label. Magika.NET does not run those rules.</summary>
    public bool RulesSupport { get; }

    public override string ToString() => Label;
}
