namespace Magika;

/// <summary>How strict the model must be before its label is kept.</summary>
public enum PredictionMode
{
    /// <summary>Keep the model label only when it clears that label's high-confidence threshold.</summary>
    HighConfidence,

    /// <summary>Keep the model label when its score clears the medium threshold (0.5 for standard_v3_3).</summary>
    MediumConfidence,

    /// <summary>Always keep the model's best label, after the overwrite map.</summary>
    BestGuess,
}
