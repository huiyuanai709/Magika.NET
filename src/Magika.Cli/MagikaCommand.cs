using System.Globalization;
using System.Text;
using Magika;

namespace Magika.Cli;

/// <summary>
/// Command-line entry point aligned with the upstream Magika Rust CLI.
/// Default text is <c>path: description (group)</c>. <c>--json</c> is a pretty JSON array and
/// <c>--jsonl</c> is one compact object per line, both using the upstream <c>path</c> / <c>result</c> schema.
/// </summary>
internal static class MagikaCommand
{
    internal static Task<int> RunAsync(string[] args) =>
        RunAsync(args, Console.Out, Console.Error, stdin: null);

    internal static async Task<int> RunAsync(string[] args, TextWriter stdout, TextWriter stderr, Stream? stdin)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(stdout);
        ArgumentNullException.ThrowIfNull(stderr);

        if (!TryParse(args, out CliOptions options, out string? parseError))
        {
            stderr.Write(parseError);
            stderr.Write('\n');
            return 2;
        }

        if (options.Help)
        {
            Write(stdout, HelpText);
            return 0;
        }

        if (options.Version)
        {
            WriteVersion(stdout);
            return 0;
        }

        if (options.Paths.Count == 0)
        {
            Write(stdout, HelpText);
            return 2;
        }

        List<string> expanded = [];
        foreach (string path in options.Paths)
        {
            expanded.AddRange(ExpandArgument(path));
        }

        if (expanded.Count(static path => path == "-") > 1)
        {
            stderr.Write("only one path can be the standard input\n");
            return 2;
        }

        List<Item> items = Collect(expanded, options.Recursive, options.FollowSymlinks);
        bool needsModel = false;
        foreach (Item item in items)
        {
            if (item.Content || item.Stdin)
            {
                needsModel = true;
                break;
            }
        }

        if (needsModel)
        {
            MagikaSession session;
            try
            {
                session = MagikaSession.Create(new MagikaOptions
                {
                    PredictionMode = options.PredictionMode,
                    FollowSymlinks = options.FollowSymlinks,
                    BatchSize = options.BatchSize,
                    InferenceWorkers = options.Workers,
                });
            }
            catch (ArgumentOutOfRangeException ex)
            {
                stderr.Write(ex.Message);
                stderr.Write('\n');
                return 2;
            }

            await IdentifyAsync(session, items, stdin).ConfigureAwait(false);
        }

        bool anyError = false;
        foreach (Item item in items)
        {
            if (item.Error is not null)
            {
                anyError = true;
            }
        }

        try
        {
            WriteReport(stdout, options, items);
        }
        catch (IOException ex) when (IsBrokenPipe(ex))
        {
        }

        return anyError ? 1 : 0;
    }

    internal static string SelectFormat(bool outputScore, bool mimeType, bool label, string? custom)
    {
        if (custom is not null)
        {
            return custom;
        }

        string body = mimeType ? "%m" : label ? "%l" : "%d (%g)";
        return outputScore ? "%p: " + body + "%b %S" : "%p: " + body + "%b";
    }

    internal static string FormatText(string format, string path, MagikaResult? result, Exception? error, bool color)
    {
        View view = View.From(path, result, error);
        return Colorize(Apply(format, view), view.Group, view.IsError, color);
    }

    internal static string FormatDocument(string path, MagikaResult? result, Exception? error, bool pretty)
    {
        var builder = new StringBuilder();
        if (!pretty)
        {
            builder.Append("{\"path\":");
            AppendJsonString(builder, path);
            builder.Append(",\"result\":");
            AppendJsonResult(builder, result, error, pretty: false, indent: 0);
            builder.Append('}');
            return builder.ToString();
        }

        builder.Append("{\n  \"path\": ");
        AppendJsonString(builder, path);
        builder.Append(",\n  \"result\": ");
        AppendJsonResult(builder, result, error, pretty: true, indent: 2);
        builder.Append("\n}");
        return builder.ToString();
    }

    internal static List<string> ExpandArgument(string path)
    {
        if (!ContainsWildcard(path) || LiteralExists(path))
        {
            return [path];
        }

        try
        {
            var matches = new List<string>();
            Glob(path, matches);
            if (matches.Count == 0)
            {
                return [path];
            }

            matches.Sort(StringComparer.Ordinal);
            return matches;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return [path];
        }
    }

    private static async Task IdentifyAsync(MagikaSession session, List<Item> items, Stream? stdin)
    {
        foreach (Item item in items)
        {
            if (!item.Stdin)
            {
                continue;
            }

            Stream stream = stdin ?? Console.OpenStandardInput();
            try
            {
                item.Result = await session.IdentifyAsync(stream, "-").ConfigureAwait(false);
            }
            catch (Exception ex) when (IsReportable(ex))
            {
                item.Error = Unwrap(ex);
            }
        }

        var indexes = new List<int>();
        for (int i = 0; i < items.Count; i++)
        {
            if (items[i].Content)
            {
                indexes.Add(i);
            }
        }

        if (indexes.Count == 0)
        {
            return;
        }

        string[] paths = new string[indexes.Count];
        for (int i = 0; i < indexes.Count; i++)
        {
            paths[i] = items[indexes[i]].Path;
        }

        var results = new MagikaResult?[paths.Length];
        var errors = new Exception?[paths.Length];
        await FillAsync(session, paths, 0, paths.Length, results, errors).ConfigureAwait(false);
        for (int i = 0; i < indexes.Count; i++)
        {
            items[indexes[i]].Result = results[i];
            items[indexes[i]].Error = errors[i];
        }
    }

    /// <summary>
    /// Scores the whole list with <see cref="MagikaSession.IdentifyPathsAsync"/> so reads stay streamed.
    /// A read error fails that call without partial results, so the list is split until the bad path is isolated.
    /// </summary>
    private static async Task FillAsync(
        MagikaSession session,
        string[] paths,
        int start,
        int count,
        MagikaResult?[] results,
        Exception?[] errors)
    {
        if (count == 0)
        {
            return;
        }

        try
        {
            if (count == 1)
            {
                results[start] = await session.IdentifyPathAsync(paths[start]).ConfigureAwait(false);
                return;
            }

            string[] slice = start == 0 && count == paths.Length ? paths : paths[start..(start + count)];
            MagikaResult[] batch = await session.IdentifyPathsAsync(slice).ConfigureAwait(false);
            for (int i = 0; i < count; i++)
            {
                results[start + i] = batch[i];
            }
        }
        catch (Exception ex)
        {
            Exception unwrapped = Unwrap(ex);
            if (count == 1)
            {
                if (!IsReportable(unwrapped))
                {
                    throw;
                }

                errors[start] = unwrapped;
                return;
            }

            if (!IsReportable(unwrapped) && ex is not AggregateException)
            {
                throw;
            }

            int mid = count / 2;
            await FillAsync(session, paths, start, mid, results, errors).ConfigureAwait(false);
            await FillAsync(session, paths, start + mid, count - mid, results, errors).ConfigureAwait(false);
        }
    }

    private static List<Item> Collect(List<string> paths, bool recursive, bool follow)
    {
        var items = new List<Item>();
        var stack = new Stack<Frame>();
        for (int i = paths.Count - 1; i >= 0; i--)
        {
            stack.Push(Frame.Path(paths[i]));
        }

        var ancestors = new HashSet<string>(PathComparer);
        while (stack.Count > 0)
        {
            Frame frame = stack.Pop();
            if (frame.Leave is not null)
            {
                ancestors.Remove(frame.Leave);
                continue;
            }

            string path = frame.PathValue!;
            if (path == "-")
            {
                items.Add(Item.StandardInput());
                continue;
            }

            try
            {
                Visit(path, recursive, follow, items, stack, ancestors);
            }
            catch (Exception ex) when (IsReportable(ex))
            {
                items.Add(Item.Failed(path, Unwrap(ex)));
            }
        }

        return items;
    }

    private static void Visit(
        string path,
        bool recursive,
        bool follow,
        List<Item> items,
        Stack<Frame> stack,
        HashSet<string> ancestors)
    {
        FileAttributes attributes = File.GetAttributes(path);
        bool link = (attributes & FileAttributes.ReparsePoint) != 0;
        if (link && !follow)
        {
            items.Add(Item.ForContent(path));
            return;
        }

        if (Directory.Exists(path))
        {
            if (!recursive)
            {
                items.Add(Item.ForContent(path));
                return;
            }

            string canonical = Canonicalize(path);
            if (!ancestors.Add(canonical))
            {
                items.Add(Item.Failed(path, new PathReport("Directory cycle", "directory_cycle")));
                return;
            }

            stack.Push(Frame.PopAncestor(canonical));
            string[] children;
            try
            {
                children = Directory.GetFileSystemEntries(path);
            }
            catch (Exception ex) when (IsReportable(ex))
            {
                items.Add(Item.Failed(path, Unwrap(ex)));
                return;
            }

            Array.Sort(children, StringComparer.Ordinal);
            for (int i = children.Length - 1; i >= 0; i--)
            {
                stack.Push(Frame.Path(children[i]));
            }

            return;
        }

        items.Add(Item.ForContent(path));
    }

    private static void WriteReport(TextWriter stdout, CliOptions options, List<Item> items)
    {
        if (options.Json)
        {
            stdout.Write('[');
            bool any = false;
            foreach (Item item in items)
            {
                if (any)
                {
                    stdout.Write(',');
                }

                any = true;
                string pretty = FormatDocument(item.Path, item.Result, item.Error, pretty: true);
                foreach (string line in pretty.Split('\n'))
                {
                    stdout.Write("\n  ");
                    stdout.Write(line);
                }
            }

            if (any)
            {
                stdout.Write('\n');
            }

            stdout.Write("]\n");
            return;
        }

        if (options.Jsonl)
        {
            foreach (Item item in items)
            {
                stdout.Write(FormatDocument(item.Path, item.Result, item.Error, pretty: false));
                stdout.Write('\n');
            }

            return;
        }

        string format = SelectFormat(options.OutputScore, options.MimeType, options.Label, options.Format);
        bool color = UseColor(options.Colors, stdout);
        foreach (Item item in items)
        {
            stdout.Write(FormatText(format, item.Path, item.Result, item.Error, color));
            stdout.Write('\n');
        }
    }

    private static bool UseColor(bool? colors, TextWriter stdout)
    {
        if (colors == true)
        {
            return true;
        }

        if (colors == false)
        {
            return false;
        }

        if (!ReferenceEquals(stdout, Console.Out) || Console.IsOutputRedirected)
        {
            return false;
        }

        return string.IsNullOrEmpty(Environment.GetEnvironmentVariable("NO_COLOR"));
    }

    private static string Colorize(string text, string group, bool error, bool color)
    {
        if (!color)
        {
            return text;
        }

        if (error)
        {
            return "\u001b[1;31m" + text + "\u001b[0m";
        }

        string sequence = group switch
        {
            "application" => "38;2;244;63;94",
            "archive" => "38;2;245;158;11",
            "audio" => "38;2;132;204;22",
            "code" => "38;2;139;92;246",
            "document" => "38;2;59;130;246",
            "executable" => "38;2;236;72;153",
            "image" => "38;2;6;182;212",
            "video" => "38;2;16;185;129",
            _ => "1;38;2;204;204;204",
        };
        return "\u001b[" + sequence + "m" + text + "\u001b[0m";
    }

    private static string Apply(string format, View view)
    {
        var builder = new StringBuilder();
        for (int i = 0; i < format.Length; i++)
        {
            char current = format[i];
            if (current != '%')
            {
                builder.Append(current);
                continue;
            }

            if (i + 1 >= format.Length)
            {
                break;
            }

            char code = format[++i];
            switch (code)
            {
                case 'p':
                    builder.Append(view.Path);
                    break;
                case 'l':
                    builder.Append(view.Label);
                    break;
                case 'd':
                    builder.Append(view.Description);
                    break;
                case 'g':
                    builder.Append(view.Group);
                    break;
                case 'm':
                    builder.Append(view.MimeType);
                    break;
                case 'e':
                    AppendExtensions(builder, view.Extensions);
                    break;
                case 's':
                    builder.Append(view.Score.ToString("0.00", CultureInfo.InvariantCulture));
                    break;
                case 'S':
                    builder.Append(Math.Truncate(100.0 * view.Score).ToString("0", CultureInfo.InvariantCulture));
                    builder.Append('%');
                    break;
                case 'b':
                    builder.Append(view.Overrule);
                    break;
                default:
                    builder.Append(code);
                    break;
            }
        }

        return builder.ToString();
    }

    private static void AppendExtensions(StringBuilder builder, IReadOnlyList<string> extensions)
    {
        builder.Append('[');
        for (int i = 0; i < extensions.Count; i++)
        {
            if (i != 0)
            {
                builder.Append(',');
            }

            builder.Append(extensions[i]);
        }

        builder.Append(']');
    }

    private static string Overrule(MagikaResult result)
    {
        if (result.ModelContentType is null)
        {
            return "";
        }

        string? reason = result.OverwriteReason switch
        {
            OverwriteReason.LowConfidence => "Low-confidence",
            OverwriteReason.RulesVeto => "Rules-veto",
            _ => null,
        };
        if (reason is null)
        {
            return "";
        }

        ContentTypeInfo best = result.ModelContentType;
        string score = result.Score.ToString("0.000", CultureInfo.InvariantCulture);
        return $" [{reason} model best-guess: {best.Description} ({best.Group}), score={score}]";
    }

    private static void AppendJsonResult(StringBuilder builder, MagikaResult? result, Exception? error, bool pretty, int indent)
    {
        if (error is not null || result is null)
        {
            Exception reported = error ?? new InvalidOperationException("Missing identification result.");
            if (!pretty)
            {
                builder.Append("{\"status\":");
                AppendJsonString(builder, StatusOf(reported));
                builder.Append('}');
                return;
            }

            string pad = new string(' ', indent);
            string inner = new string(' ', indent + 2);
            builder.Append("{\n");
            builder.Append(inner).Append("\"status\": ");
            AppendJsonString(builder, StatusOf(reported));
            builder.Append('\n').Append(pad).Append('}');
            return;
        }

        ContentTypeInfo output = result.ContentType;
        ContentTypeInfo dl = result.ModelContentType ?? ContentTypes.Get("undefined");
        float score = result.ModelContentType is null ? 1f : result.Score;
        if (!pretty)
        {
            builder.Append("{\"status\":\"ok\",\"value\":{\"dl\":");
            AppendJsonType(builder, dl, pretty: false, indent: 0);
            builder.Append(",\"output\":");
            AppendJsonType(builder, output, pretty: false, indent: 0);
            builder.Append(",\"score\":").Append(FormatJsonScore(score)).Append("}}");
            return;
        }

        string padOk = new string(' ', indent);
        string innerOk = new string(' ', indent + 2);
        builder.Append("{\n");
        builder.Append(innerOk).Append("\"status\": \"ok\",\n");
        builder.Append(innerOk).Append("\"value\": {\n");
        string field = new string(' ', indent + 4);
        builder.Append(field).Append("\"dl\": ");
        AppendJsonType(builder, dl, pretty: true, indent: indent + 4);
        builder.Append(",\n");
        builder.Append(field).Append("\"output\": ");
        AppendJsonType(builder, output, pretty: true, indent: indent + 4);
        builder.Append(",\n");
        builder.Append(field).Append("\"score\": ").Append(FormatJsonScore(score)).Append('\n');
        builder.Append(innerOk).Append("}\n");
        builder.Append(padOk).Append('}');
    }

    private static void AppendJsonType(StringBuilder builder, ContentTypeInfo type, bool pretty, int indent)
    {
        if (!pretty)
        {
            builder.Append("{\"description\":");
            AppendJsonString(builder, type.Description);
            builder.Append(",\"extensions\":");
            AppendJsonExtensions(builder, type.Extensions, pretty: false, indent: 0);
            builder.Append(",\"group\":");
            AppendJsonString(builder, type.Group);
            builder.Append(",\"is_text\":").Append(type.IsText ? "true" : "false");
            builder.Append(",\"label\":");
            AppendJsonString(builder, type.Label);
            builder.Append(",\"mime_type\":");
            AppendJsonString(builder, type.MimeType);
            builder.Append('}');
            return;
        }

        string pad = new string(' ', indent);
        string inner = new string(' ', indent + 2);
        builder.Append("{\n");
        builder.Append(inner).Append("\"description\": ");
        AppendJsonString(builder, type.Description);
        builder.Append(",\n");
        builder.Append(inner).Append("\"extensions\": ");
        AppendJsonExtensions(builder, type.Extensions, pretty: true, indent: indent + 2);
        builder.Append(",\n");
        builder.Append(inner).Append("\"group\": ");
        AppendJsonString(builder, type.Group);
        builder.Append(",\n");
        builder.Append(inner).Append("\"is_text\": ").Append(type.IsText ? "true" : "false");
        builder.Append(",\n");
        builder.Append(inner).Append("\"label\": ");
        AppendJsonString(builder, type.Label);
        builder.Append(",\n");
        builder.Append(inner).Append("\"mime_type\": ");
        AppendJsonString(builder, type.MimeType);
        builder.Append('\n').Append(pad).Append('}');
    }

    private static void AppendJsonExtensions(StringBuilder builder, IReadOnlyList<string> extensions, bool pretty, int indent)
    {
        if (!pretty || extensions.Count == 0)
        {
            builder.Append('[');
            for (int i = 0; i < extensions.Count; i++)
            {
                if (i != 0)
                {
                    builder.Append(',');
                }

                AppendJsonString(builder, extensions[i]);
            }

            builder.Append(']');
            return;
        }

        builder.Append("[\n");
        string inner = new string(' ', indent + 2);
        for (int i = 0; i < extensions.Count; i++)
        {
            builder.Append(inner);
            AppendJsonString(builder, extensions[i]);
            if (i + 1 != extensions.Count)
            {
                builder.Append(',');
            }

            builder.Append('\n');
        }

        builder.Append(new string(' ', indent)).Append(']');
    }

    private static void AppendJsonString(StringBuilder builder, string value)
    {
        builder.Append('"');
        foreach (char character in value)
        {
            switch (character)
            {
                case '\\':
                    builder.Append("\\\\");
                    break;
                case '"':
                    builder.Append("\\\"");
                    break;
                case '\n':
                    builder.Append("\\n");
                    break;
                case '\r':
                    builder.Append("\\r");
                    break;
                case '\t':
                    builder.Append("\\t");
                    break;
                default:
                    if (character < 0x20)
                    {
                        builder.Append("\\u");
                        builder.Append(((int)character).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        builder.Append(character);
                    }

                    break;
            }
        }

        builder.Append('"');
    }

    /// <summary>
    /// Upstream stores the JSON document in <c>serde_json::Value</c>, whose map is a <c>BTreeMap</c>
    /// unless <c>preserve_order</c> is on. Keys therefore come out sorted. Scores are truncated to
    /// three decimal places before serialization; this writer prints that decimal without f32 noise.
    /// </summary>
    private static string FormatJsonScore(float score)
    {
        double truncated = Math.Truncate((double)score * 1000.0) / 1000.0;
        return truncated.ToString("0.0##", CultureInfo.InvariantCulture);
    }

    private static string StatusOf(Exception error)
    {
        if (error is PathReport report)
        {
            return report.Status;
        }

        if (error is FileNotFoundException or DirectoryNotFoundException)
        {
            return "file_does_not_exist";
        }

        if (error is UnauthorizedAccessException || IsPermissionDenied(error))
        {
            return "permission_error";
        }

        if (error.Message == "Directory cycle")
        {
            return "directory_cycle";
        }

        return "unknown";
    }

    private static bool IsPermissionDenied(Exception error) =>
        error is IOException
        && (error.Message.Contains("Permission denied", StringComparison.OrdinalIgnoreCase)
            || error.Message.Contains("Access to the path", StringComparison.OrdinalIgnoreCase)
            || error.Message.Contains("is denied", StringComparison.OrdinalIgnoreCase));

    private static void Glob(string pattern, List<string> matches)
    {
        string normalized = pattern.Replace(Path.AltDirectorySeparatorChar, Path.DirectorySeparatorChar);
        bool rooted = Path.IsPathRooted(normalized);
        string root = rooted ? Path.GetPathRoot(normalized) ?? "" : "";
        string rest = rooted ? normalized[root.Length..] : normalized;
        string[] parts = rest.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        int wild = 0;
        while (wild < parts.Length && !ContainsWildcard(parts[wild]))
        {
            wild++;
        }

        string baseDir = root;
        for (int i = 0; i < wild; i++)
        {
            baseDir = baseDir.Length == 0 ? parts[i] : Path.Combine(baseDir, parts[i]);
        }

        if (baseDir.Length == 0)
        {
            baseDir = ".";
        }

        if (wild >= parts.Length)
        {
            return;
        }

        var seen = new HashSet<string>(PathComparer);
        Match(baseDir, parts, wild, seen, matches);
    }

    private static void Match(string directory, string[] parts, int index, HashSet<string> seen, List<string> matches)
    {
        string part = parts[index];
        bool last = index == parts.Length - 1;
        if (!ContainsWildcard(part))
        {
            string next = Path.Combine(directory, part);
            if (last)
            {
                if (LiteralExists(next))
                {
                    matches.Add(next);
                }

                return;
            }

            if (Directory.Exists(next))
            {
                Enter(next, parts, index + 1, seen, matches);
            }

            return;
        }

        if (!Directory.Exists(directory))
        {
            return;
        }

        foreach (string entry in Directory.EnumerateFileSystemEntries(directory))
        {
            if (!SegmentMatch(Path.GetFileName(entry), part))
            {
                continue;
            }

            if (last)
            {
                matches.Add(entry);
            }
            else if (Directory.Exists(entry))
            {
                Enter(entry, parts, index + 1, seen, matches);
            }
        }
    }

    private static void Enter(string directory, string[] parts, int index, HashSet<string> seen, List<string> matches)
    {
        string canonical = Canonicalize(directory);
        if (!seen.Add(canonical))
        {
            return;
        }

        try
        {
            Match(directory, parts, index, seen, matches);
        }
        finally
        {
            seen.Remove(canonical);
        }
    }

    private static bool SegmentMatch(string name, string pattern) =>
        SegmentMatch(name, 0, pattern, 0);

    private static bool SegmentMatch(string name, int nameIndex, string pattern, int patternIndex)
    {
        while (patternIndex < pattern.Length)
        {
            if (pattern[patternIndex] == '*')
            {
                while (patternIndex < pattern.Length && pattern[patternIndex] == '*')
                {
                    patternIndex++;
                }

                if (patternIndex == pattern.Length)
                {
                    return true;
                }

                for (int start = nameIndex; start <= name.Length; start++)
                {
                    if (SegmentMatch(name, start, pattern, patternIndex))
                    {
                        return true;
                    }
                }

                return false;
            }

            if (nameIndex >= name.Length)
            {
                return false;
            }

            if (pattern[patternIndex] != '?' && !GlobCharEquals(pattern[patternIndex], name[nameIndex]))
            {
                return false;
            }

            patternIndex++;
            nameIndex++;
        }

        return nameIndex == name.Length;
    }

    private static bool GlobCharEquals(char pattern, char name) =>
        OperatingSystem.IsWindows()
            ? char.ToUpperInvariant(pattern) == char.ToUpperInvariant(name)
            : pattern == name;

    private static bool ContainsWildcard(string path) =>
        path.IndexOf('*') >= 0 || path.IndexOf('?') >= 0;

    private static bool LiteralExists(string path) => File.Exists(path) || Directory.Exists(path);

    private static string Canonicalize(string path)
    {
        string full = Path.GetFullPath(path);
        string root = Path.GetPathRoot(full) ?? "";
        string relative = root.Length == 0 ? full : full[root.Length..];
        string current = root;
        foreach (string segment in relative.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries))
        {
            current = current.Length == 0 ? segment : Path.Combine(current, segment);
            string? linked = TryFinalTarget(current);
            if (linked is not null)
            {
                current = Path.TrimEndingDirectorySeparator(linked);
            }
        }

        if (current.Length == 0)
        {
            return full;
        }

        return Path.TrimEndingDirectorySeparator(current);
    }

    private static string? TryFinalTarget(string path)
    {
        try
        {
            return File.ResolveLinkTarget(path, returnFinalTarget: true)?.FullName;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return null;
        }
    }

    private static bool TryParse(string[] args, out CliOptions options, out string? error)
    {
        options = new CliOptions();
        error = null;
        bool ended = false;
        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            if (!ended && arg == "--")
            {
                ended = true;
                continue;
            }

            if (ended || arg == "-" || !arg.StartsWith('-'))
            {
                options.Paths.Add(arg);
                continue;
            }

            if (arg is "-h" or "--help")
            {
                options.Help = true;
                return true;
            }

            if (arg is "-V" or "--version")
            {
                options.Version = true;
                return true;
            }

            if (!arg.StartsWith("--"))
            {
                if (!TryParseCluster(arg, options, out string? clusterError))
                {
                    error = clusterError;
                    return false;
                }

                if (options.Help || options.Version)
                {
                    return true;
                }

                continue;
            }

            if (!TryParseLong(args, ref i, arg, options, out string? longError))
            {
                error = longError;
                return false;
            }
        }

        if (error is not null)
        {
            return false;
        }

        if (options.BatchSize is < 1 or > 64)
        {
            error = "Missing or invalid value for --batch-size (1..64).";
            return false;
        }

        if (options.Workers < 0)
        {
            error = "Missing or invalid value for --workers. Use 0 for the automatic lane count, or a positive count.";
            return false;
        }

        int formats = (options.Json ? 1 : 0) + (options.Jsonl ? 1 : 0) + (options.Format is not null ? 1 : 0);
        if (formats > 1)
        {
            error = "The options --json, --jsonl, and --format cannot be combined.";
            return false;
        }

        bool modifiers = options.OutputScore || options.MimeType || options.Label;
        if (modifiers && formats > 0)
        {
            error = "The options -s/--output-score, -i/--mime-type, and -l/--label cannot be combined with --json, --jsonl, or --format.";
            return false;
        }

        if (options.MimeType && options.Label)
        {
            error = "The options -i/--mime-type and -l/--label cannot be combined.";
            return false;
        }

        return true;
    }

    private static bool TryParseCluster(string arg, CliOptions options, out string? error)
    {
        for (int i = 1; i < arg.Length; i++)
        {
            switch (arg[i])
            {
                case 'h':
                    options.Help = true;
                    error = null;
                    return true;
                case 'V':
                    options.Version = true;
                    error = null;
                    return true;
                case 'r':
                    options.Recursive = true;
                    break;
                case 's':
                    options.OutputScore = true;
                    break;
                case 'i':
                    options.MimeType = true;
                    break;
                case 'l':
                    options.Label = true;
                    break;
                default:
                    error = $"Unknown option '-{arg[i]}'.";
                    return false;
            }
        }

        error = null;
        return true;
    }

    private static bool TryParseLong(string[] args, ref int index, string arg, CliOptions options, out string? error)
    {
        string name = arg;
        string? inline = null;
        int equals = arg.IndexOf('=', StringComparison.Ordinal);
        if (equals >= 0)
        {
            name = arg[..equals];
            inline = arg[(equals + 1)..];
        }

        switch (name)
        {
            case "--recursive":
                return Flag(name, inline, () => options.Recursive = true, out error);
            case "--no-dereference":
                return Flag(name, inline, () => options.FollowSymlinks = false, out error);
            case "--colors":
                return SetColors(options, name, inline, enable: true, out error);
            case "--no-colors":
                return SetColors(options, name, inline, enable: false, out error);
            case "--output-score":
                return Flag(name, inline, () => options.OutputScore = true, out error);
            case "--mime-type":
                return Flag(name, inline, () => options.MimeType = true, out error);
            case "--label":
                return Flag(name, inline, () => options.Label = true, out error);
            case "--json":
                return Flag(name, inline, () => options.Json = true, out error);
            case "--jsonl":
                return Flag(name, inline, () => options.Jsonl = true, out error);
            case "--format":
                if (!TryValue(args, ref index, name, inline, out string format, out error))
                {
                    return false;
                }

                options.Format = format;
                return true;
            case "--prediction-mode":
                if (!TryValue(args, ref index, name, inline, out string modeText, out error))
                {
                    return false;
                }

                if (!TryParseMode(modeText, out PredictionMode mode))
                {
                    error = $"Unknown prediction mode '{modeText}'.";
                    return false;
                }

                options.PredictionMode = mode;
                return true;
            case "--batch-size":
                if (!TryValue(args, ref index, name, inline, out string batchText, out error))
                {
                    return false;
                }

                if (!int.TryParse(batchText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int batchSize))
                {
                    error = "Missing or invalid value for --batch-size (1..64).";
                    return false;
                }

                options.BatchSize = batchSize;
                return true;
            case "--workers":
                if (!TryValue(args, ref index, name, inline, out string workerText, out error))
                {
                    return false;
                }

                if (!int.TryParse(workerText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int workers))
                {
                    error = "Missing or invalid value for --workers. Use 0 for the automatic lane count, or a positive count.";
                    return false;
                }

                options.Workers = workers;
                return true;
            default:
                error = $"Unknown option '{name}'.";
                return false;
        }
    }

    private static bool SetColors(CliOptions options, string name, string? inline, bool enable, out string? error)
    {
        if (inline is not null)
        {
            error = $"Option '{name}' does not take a value.";
            return false;
        }

        if (options.Colors is not null && options.Colors != enable)
        {
            error = "The options --colors and --no-colors cannot be combined.";
            return false;
        }

        options.Colors = enable;
        error = null;
        return true;
    }

    private static bool Flag(string name, string? inline, Action set, out string? error)
    {
        if (inline is not null)
        {
            error = $"Option '{name}' does not take a value.";
            return false;
        }

        set();
        error = null;
        return true;
    }

    private static bool TryValue(string[] args, ref int index, string name, string? inline, out string value, out string? error)
    {
        if (inline is not null)
        {
            value = inline;
            error = null;
            return true;
        }

        if (index + 1 >= args.Length)
        {
            value = "";
            error = $"Missing value for {name}.";
            return false;
        }

        value = args[++index];
        error = null;
        return true;
    }

    private static bool TryParseMode(string text, out PredictionMode mode)
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

    private static void WriteVersion(TextWriter stdout)
    {
        Write(stdout, $"magika {typeof(MagikaSession).Assembly.GetName().Version} (standard_v3_3)");
        if (GC.GetConfigurationVariables().ContainsKey("SatoriGC"))
        {
            Write(stdout, "gc: satori");
        }
    }

    private static void Write(TextWriter writer, string text)
    {
        writer.Write(text);
        if (!text.EndsWith('\n'))
        {
            writer.Write('\n');
        }
    }

    private static bool IsReportable(Exception exception)
    {
        Exception unwrapped = Unwrap(exception);
        return unwrapped is IOException or UnauthorizedAccessException or ArgumentException or PathReport;
    }

    private static Exception Unwrap(Exception exception)
    {
        if (exception is AggregateException aggregate && aggregate.InnerExceptions.Count == 1 && aggregate.InnerException is not null)
        {
            return Unwrap(aggregate.InnerException);
        }

        return exception;
    }

    private static bool IsBrokenPipe(IOException exception)
    {
        int code = exception.HResult & 0xFFFF;
        return code is 32 or 109 or 232;
    }

    private static StringComparer PathComparer =>
        OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

    private const string HelpText =
        """
        magika — identify file content types

        Usage: magika [OPTIONS] [PATH]...

        Arguments:
          [PATH]...
                  List of paths to the files to analyze.
                  Use a dash (-) to read from standard input (can only be used once).

        Options:
          -r, --recursive
                  Identifies files within directories instead of identifying the directory itself

              --no-dereference
                  Identifies symbolic links as is instead of identifying their content by following them

              --colors
                  Prints with colors regardless of terminal support

              --no-colors
                  Prints without colors regardless of terminal support

          -s, --output-score
                  Prints the prediction score in addition to the content type

          -i, --mime-type
                  Prints the MIME type instead of the content type description

          -l, --label
                  Prints a simple label instead of the content type description

              --json
                  Prints in JSON format

              --jsonl
                  Prints in JSONL format

              --format <CUSTOM>
                  Prints using a custom format.

                  The following placeholders are supported:

                    %p  The file path
                    %l  The unique label identifying the content type
                    %d  The description of the content type
                    %g  The group of the content type
                    %m  The MIME type of the content type
                    %e  Possible file extensions for the content type
                    %s  The score of the content type for the file
                    %S  The score of the content type for the file in percent
                    %b  The model output if overruled (empty otherwise)
                    %%  A literal %

              --prediction-mode <MODE>
                  high (default), medium, or best-guess. Magika.NET only.

              --batch-size <N>
                  Feature rows per forward pass, from 1 to 64. Default 8. Magika.NET only.

              --workers <N>
                  Inference lanes. 0 (default) is automatic. Magika.NET only.

          -h, --help
                  Print help

          -V, --version
                  Print version

        The default line is "path: description (group)", matching the upstream Magika CLI.
        -s adds a truncated percent score. -l prints the label and -i prints the MIME type.
        -s, -i, and -l cannot be combined with --json, --jsonl, or --format.
        -i and -l cannot be combined. --json, --jsonl, and --format cannot be combined.
        --colors and --no-colors cannot be combined.

        Colors follow the content-type group when standard output is a terminal.
        NO_COLOR disables them. --colors forces them and --no-colors disables them.

        If an argument contains * or ? and that exact path does not exist, it is expanded
        as a file-name pattern. An existing path is left as written, even when the name
        contains those characters.

        Per-file errors are reported in the output and the process exits 1 after the rest
        of the paths. Exit code 2 means the arguments were invalid.

        The model is Magika standard_v3_3, executed by a managed ONNX interpreter.
        There is no native ONNX runtime.
        Several files are read concurrently, then scored in batches (default batch size 8).
        --workers 0 uses one lane for this model so its convolution can use every processor.
        An explicit count is not capped. Each lane is its own managed session.
        """;

    private readonly struct View(
        string path,
        string label,
        string description,
        string group,
        string mimeType,
        IReadOnlyList<string> extensions,
        float score,
        string overrule,
        bool isError)
    {
        public string Path { get; } = path;

        public string Label { get; } = label;

        public string Description { get; } = description;

        public string Group { get; } = group;

        public string MimeType { get; } = mimeType;

        public IReadOnlyList<string> Extensions { get; } = extensions;

        public float Score { get; } = score;

        public string Overrule { get; } = overrule;

        public bool IsError { get; } = isError;

        public static View From(string path, MagikaResult? result, Exception? error)
        {
            if (error is not null || result is null)
            {
                string message = error?.Message ?? "Missing identification result.";
                return new View(path, "error", message, "error", "error", [], score: 1f, overrule: "", isError: true);
            }

            ContentTypeInfo type = result.ContentType;
            return new View(
                path,
                type.Label,
                type.Description,
                type.Group,
                type.MimeType,
                type.Extensions,
                result.Score,
                MagikaCommand.Overrule(result),
                isError: false);
        }
    }

    private readonly struct Frame
    {
        private Frame(string? path, string? leave)
        {
            PathValue = path;
            Leave = leave;
        }

        public string? PathValue { get; }

        public string? Leave { get; }

        public static Frame Path(string path) => new(path, null);

        public static Frame PopAncestor(string canonical) => new(null, canonical);
    }

    private sealed class Item
    {
        private Item(string path, bool stdin, bool content, Exception? error)
        {
            Path = path;
            Stdin = stdin;
            Content = content;
            Error = error;
        }

        public string Path { get; }

        public bool Stdin { get; }

        public bool Content { get; }

        public MagikaResult? Result { get; set; }

        public Exception? Error { get; set; }

        public static Item StandardInput() => new("-", stdin: true, content: false, error: null);

        public static Item ForContent(string path) => new(path, stdin: false, content: true, error: null);

        public static Item Failed(string path, Exception error) => new(path, stdin: false, content: false, error);
    }

    private sealed class PathReport : Exception
    {
        public PathReport(string message, string status)
            : base(message)
        {
            Status = status;
        }

        public string Status { get; }
    }

    private sealed class CliOptions
    {
        public bool Help { get; set; }

        public bool Version { get; set; }

        public bool Recursive { get; set; }

        public bool FollowSymlinks { get; set; } = true;

        public bool? Colors { get; set; }

        public bool OutputScore { get; set; }

        public bool MimeType { get; set; }

        public bool Label { get; set; }

        public bool Json { get; set; }

        public bool Jsonl { get; set; }

        public string? Format { get; set; }

        public PredictionMode PredictionMode { get; set; } = PredictionMode.HighConfidence;

        public int BatchSize { get; set; } = MagikaOptions.Default.BatchSize;

        public int Workers { get; set; } = MagikaOptions.Default.InferenceWorkers;

        public List<string> Paths { get; } = [];
    }
}
