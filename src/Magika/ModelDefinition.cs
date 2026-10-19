using System.Text.Json;
using System.Text.Json.Serialization;

namespace Magika;

internal sealed class ModelDefinition
{
    public const string Name = "standard_v3_3";
    public const int MajorVersion = 3;

    public required int BegSize { get; init; }
    public required int EndSize { get; init; }
    public required int BlockSize { get; init; }
    public required int PaddingToken { get; init; }
    public required int MinFileSizeForDl { get; init; }
    public required float MediumThreshold { get; init; }
    public required string[] TrainingLabels { get; init; }
    public required string[] OutputLabels { get; init; }
    public required float[] HighThresholds { get; init; }

    public int FeaturesSize => BegSize + EndSize;

    public static ModelDefinition LoadDefault()
    {
        using Stream stream = OpenResource("Magika.Models.standard_v3_3.config.min.json");
        ModelConfigFile file = JsonSerializer.Deserialize(stream, ModelJsonContext.Default.ModelConfigFile)
            ?? throw new InvalidDataException("Magika model config is empty.");
        if (file.MidSize != 0 || file.UseInputsAtOffsets)
        {
            throw new NotSupportedException("This build only supports beg/end Magika models (mid_size = 0, no offset inputs).");
        }

        if (file.TargetLabelsSpace is null || file.TargetLabelsSpace.Length == 0)
        {
            throw new InvalidDataException("Magika model config has no target labels.");
        }

        string[] training = file.TargetLabelsSpace;
        string[] output = new string[training.Length];
        float[] thresholds = new float[training.Length];
        IReadOnlyDictionary<string, string> overwrite = file.OverwriteMap ?? new Dictionary<string, string>();
        IReadOnlyDictionary<string, float> high = file.Thresholds ?? new Dictionary<string, float>();
        for (int i = 0; i < training.Length; i++)
        {
            string label = training[i];
            output[i] = overwrite.TryGetValue(label, out string? mapped) ? mapped : label;
            thresholds[i] = high.TryGetValue(label, out float threshold) ? threshold : 0.5f;
            _ = ContentTypes.Get(label);
            _ = ContentTypes.Get(output[i]);
        }

        return new ModelDefinition
        {
            BegSize = file.BegSize,
            EndSize = file.EndSize,
            BlockSize = file.BlockSize,
            PaddingToken = file.PaddingToken,
            MinFileSizeForDl = file.MinFileSizeForDl,
            MediumThreshold = file.MediumConfidenceThreshold,
            TrainingLabels = training,
            OutputLabels = output,
            HighThresholds = thresholds,
        };
    }

    public static Stream OpenModel() => OpenResource("Magika.Models.standard_v3_3.model.onnx");

    private static Stream OpenResource(string name)
    {
        return typeof(ModelDefinition).Assembly.GetManifestResourceStream(name)
            ?? throw new InvalidDataException($"Embedded Magika resource '{name}' is missing.");
    }
}

internal sealed class ModelConfigFile
{
    public int BegSize { get; set; }
    public int MidSize { get; set; }
    public int EndSize { get; set; }
    public bool UseInputsAtOffsets { get; set; }
    public float MediumConfidenceThreshold { get; set; }
    public int MinFileSizeForDl { get; set; }
    public int PaddingToken { get; set; }
    public int BlockSize { get; set; }
    public string[]? TargetLabelsSpace { get; set; }
    public Dictionary<string, float>? Thresholds { get; set; }
    public Dictionary<string, string>? OverwriteMap { get; set; }
}

[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    PropertyNameCaseInsensitive = false)]
[JsonSerializable(typeof(ModelConfigFile))]
internal sealed partial class ModelJsonContext : JsonSerializerContext;
