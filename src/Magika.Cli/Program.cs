using System.Globalization;
using System.Text;
using Magika;

return Run(args);

static int Run(string[] args)
{
    PredictionMode mode = PredictionMode.HighConfidence;
    bool json = false;
    bool follow = true;
    List<string> paths = [];
    for (int i = 0; i < args.Length; i++)
    {
        string arg = args[i];
        switch (arg)
        {
            case "-h":
            case "--help":
                PrintHelp();
                return 0;
            case "--version":
                Console.WriteLine($"magika {typeof(MagikaSession).Assembly.GetName().Version} ({ModelDefinitionName()})");
                if (GC.GetConfigurationVariables().ContainsKey("SatoriGC"))
                {
                    Console.WriteLine("gc: satori");
                }
                return 0;
            case "--json":
                json = true;
                break;
            case "--no-dereference":
                follow = false;
                break;
            case "--prediction-mode":
                if (i + 1 >= args.Length)
                {
                    Console.Error.WriteLine("Missing value for --prediction-mode.");
                    return 2;
                }

                if (!TryParseMode(args[++i], out mode))
                {
                    Console.Error.WriteLine($"Unknown prediction mode '{args[i]}'.");
                    return 2;
                }

                break;
            default:
                if (arg.StartsWith('-'))
                {
                    Console.Error.WriteLine($"Unknown option '{arg}'.");
                    return 2;
                }

                paths.Add(arg);
                break;
        }
    }

    if (paths.Count == 0)
    {
        PrintHelp();
        return 2;
    }

    MagikaSession session = MagikaSession.Create(new MagikaOptions
    {
        PredictionMode = mode,
        FollowSymlinks = follow,
    });

    foreach (string path in paths)
    {
        try
        {
            MagikaResult result = session.IdentifyPath(path);
            Console.WriteLine(json ? ToJson(path, result) : ToText(path, result));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FileNotFoundException)
        {
            Console.Error.WriteLine($"{path}: {ex.Message}");
            return 1;
        }
    }

    return 0;
}

static string ModelDefinitionName() => "standard_v3_3";

static bool TryParseMode(string text, out PredictionMode mode)
{
    switch (text.Replace('_', '-').ToLowerInvariant())
    {
        case "high":
        case "high-confidence":
            mode = PredictionMode.HighConfidence;
            return true;
        case "medium":
        case "medium-confidence":
            mode = PredictionMode.MediumConfidence;
            return true;
        case "best":
        case "best-guess":
            mode = PredictionMode.BestGuess;
            return true;
        default:
            mode = default;
            return false;
    }
}

static string ToText(string path, MagikaResult result)
{
    ContentTypeInfo type = result.ContentType;
    string score = result.Score.ToString("0.000", CultureInfo.InvariantCulture);
    return $"{path}: {type.Label} {score} {type.MimeType} {type.Group}";
}

static string ToJson(string path, MagikaResult result)
{
    ContentTypeInfo type = result.ContentType;
    StringBuilder builder = new();
    builder.Append("{\"path\":\"").Append(Escape(path)).Append('"');
    builder.Append(",\"label\":\"").Append(Escape(type.Label)).Append('"');
    builder.Append(",\"mime_type\":\"").Append(Escape(type.MimeType)).Append('"');
    builder.Append(",\"group\":\"").Append(Escape(type.Group)).Append('"');
    builder.Append(",\"description\":\"").Append(Escape(type.Description)).Append('"');
    builder.Append(",\"score\":").Append(result.Score.ToString("0.000000", CultureInfo.InvariantCulture));
    builder.Append(",\"is_text\":").Append(type.IsText ? "true" : "false");
    if (result.ModelContentType is not null)
    {
        builder.Append(",\"model_label\":\"").Append(Escape(result.ModelContentType.Label)).Append('"');
    }

    if (result.OverwriteReason is not null)
    {
        builder.Append(",\"overwrite_reason\":\"").Append(Reason(result.OverwriteReason.Value)).Append('"');
    }

    builder.Append('}');
    return builder.ToString();
}

static string Reason(OverwriteReason reason) => reason switch
{
    OverwriteReason.LowConfidence => "low_confidence",
    OverwriteReason.OverwriteMap => "overwrite_map",
    OverwriteReason.RulesVeto => "rules_veto",
    _ => reason.ToString(),
};

static string Escape(string value) =>
    value.Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\"", "\\\"", StringComparison.Ordinal);

static void PrintHelp()
{
    Console.WriteLine(
        """
        magika — identify file content types

        Usage:
          magika [--json] [--prediction-mode high|medium|best-guess] [--no-dereference] <file>...

        The model is Magika standard_v3_3, executed by a managed ONNX interpreter.
        """);
}
