using System.Globalization;
using System.Text;
using Magika;

return await RunAsync(args);

static async Task<int> RunAsync(string[] args)
{
    PredictionMode mode = PredictionMode.HighConfidence;
    bool json = false;
    bool follow = true;
    int batchSize = MagikaOptions.Default.BatchSize;
    int workers = MagikaOptions.Default.InferenceWorkers;
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
            case "--batch-size":
                if (i + 1 >= args.Length || !int.TryParse(args[++i], out batchSize))
                {
                    Console.Error.WriteLine("Missing or invalid value for --batch-size (1..64).");
                    return 2;
                }

                break;
            case "--workers":
                if (i + 1 >= args.Length || !int.TryParse(args[++i], out workers) || workers < 0)
                {
                    Console.Error.WriteLine("Missing or invalid value for --workers. Use 0 for the automatic lane count, or a positive count.");
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

    MagikaSession session;
    try
    {
        session = MagikaSession.Create(new MagikaOptions
        {
            PredictionMode = mode,
            FollowSymlinks = follow,
            BatchSize = batchSize,
            InferenceWorkers = workers,
        });
    }
    catch (ArgumentOutOfRangeException ex)
    {
        Console.Error.WriteLine(ex.Message);
        return 2;
    }

    if (paths.Count == 1)
    {
        return await WriteOneAsync(paths[0], session, json).ConfigureAwait(false);
    }

    try
    {
        MagikaResult[] results = await session.IdentifyPathsAsync(paths).ConfigureAwait(false);
        for (int i = 0; i < results.Length; i++)
        {
            Console.WriteLine(json ? ToJson(paths[i], results[i]) : ToText(paths[i], results[i]));
        }
    }
    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FileNotFoundException)
    {
        string path = ex is FileNotFoundException missing && !string.IsNullOrEmpty(missing.FileName)
            ? missing.FileName
            : "";
        Console.Error.WriteLine(path.Length == 0 ? ex.Message : $"{path}: {ex.Message}");
        return 1;
    }

    return 0;
}

static async Task<int> WriteOneAsync(string path, MagikaSession session, bool json)
{
    try
    {
        MagikaResult result = await session.IdentifyPathAsync(path).ConfigureAwait(false);
        Console.WriteLine(json ? ToJson(path, result) : ToText(path, result));
        return 0;
    }
    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FileNotFoundException)
    {
        Console.Error.WriteLine($"{path}: {ex.Message}");
        return 1;
    }
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
          magika [--json] [--prediction-mode high|medium|best-guess] [--no-dereference]
                 [--batch-size N] [--workers N] <file>...

        The model is Magika standard_v3_3, executed by a managed ONNX interpreter.
        Several files are read concurrently, then scored in batches (default batch size 8).
        --workers 0 is automatic: one lane for the specialized model, so its convolution can use every processor.
        An explicit count is not capped. Extra lanes split that convolution across the pool.
        Each lane is its own managed session. There is no native ONNX runtime.
        """);
}
