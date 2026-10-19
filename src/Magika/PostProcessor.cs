namespace Magika;

internal readonly struct ModelDecision
{
    public int BestIndex { get; init; }
    public string TrainingLabel { get; init; }
    public string Label { get; init; }
    public float Score { get; init; }
    public OverwriteReason? Reason { get; init; }
}

internal static class PostProcessor
{
    internal static ModelDecision Decide(ModelDefinition model, PredictionMode mode, ReadOnlySpan<float> scores)
    {
        if (scores.Length != model.TrainingLabels.Length)
        {
            throw new InvalidDataException(
                $"Model returned {scores.Length} scores, expected {model.TrainingLabels.Length}.");
        }

        int best = 0;
        float score = scores[0];
        for (int i = 1; i < scores.Length; i++)
        {
            // Ties take the later index, matching the Rust best_index helper.
            if (scores[i] >= score)
            {
                best = i;
                score = scores[i];
            }
        }

        string training = model.TrainingLabels[best];
        string mapped = model.OutputLabels[best];
        bool confident = mode switch
        {
            PredictionMode.HighConfidence => score >= model.HighThresholds[best],
            PredictionMode.MediumConfidence => score >= model.MediumThreshold,
            PredictionMode.BestGuess => true,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

        string label;
        OverwriteReason? reason;
        if (!confident)
        {
            label = ContentTypes.Get(mapped).IsText ? "txt" : "unknown";
            reason = OverwriteReason.LowConfidence;
        }
        else
        {
            label = mapped;
            reason = OverwriteReason.OverwriteMap;
        }

        if (label == training)
        {
            reason = null;
        }

        return new ModelDecision
        {
            BestIndex = best,
            TrainingLabel = training,
            Label = label,
            Score = score,
            Reason = reason,
        };
    }
}
