using System.Text.Json;
using Magika.Cli;
using Xunit;

namespace Magika.Tests;

public class CliTests
{
    [Fact]
    public async Task Help_ListsUpstreamAndMagikaFlags()
    {
        var stdout = new StringWriter();
        var stderr = new StringWriter();
        int code = await MagikaCommand.RunAsync(["-h"], stdout, stderr, stdin: null);

        Assert.Equal(0, code);
        Assert.Equal("", stderr.ToString());
        string help = stdout.ToString();
        Assert.Contains("-r, --recursive", help);
        Assert.Contains("--no-dereference", help);
        Assert.Contains("--colors", help);
        Assert.Contains("--no-colors", help);
        Assert.Contains("-s, --output-score", help);
        Assert.Contains("-i, --mime-type", help);
        Assert.Contains("-l, --label", help);
        Assert.Contains("--jsonl", help);
        Assert.Contains("--format <CUSTOM>", help);
        Assert.Contains("%S", help);
        Assert.Contains("%%", help);
        Assert.Contains("--prediction-mode <MODE>", help);
        Assert.Contains("--batch-size <N>", help);
        Assert.Contains("--workers <N>", help);
        Assert.Contains("-V, --version", help);
        Assert.Contains("path: description (group)", help);
    }

    [Fact]
    public async Task Version_AcceptsShortAndLongAliases()
    {
        var stdout = new StringWriter();
        int shortCode = await MagikaCommand.RunAsync(["-V"], stdout, new StringWriter(), stdin: null);
        string shortText = stdout.ToString();
        stdout = new StringWriter();
        int longCode = await MagikaCommand.RunAsync(["--version"], stdout, new StringWriter(), stdin: null);

        Assert.Equal(0, shortCode);
        Assert.Equal(0, longCode);
        Assert.Contains("0.1.4", shortText);
        Assert.Contains("standard_v3_3", shortText);
        Assert.Equal(shortText, stdout.ToString());
    }

    [Fact]
    public async Task NoPaths_PrintsHelpAndExits2()
    {
        var stdout = new StringWriter();
        int code = await MagikaCommand.RunAsync([], stdout, new StringWriter(), stdin: null);
        Assert.Equal(2, code);
        Assert.Contains("--jsonl", stdout.ToString());
    }

    [Theory]
    [InlineData("--json", "--jsonl")]
    [InlineData("--json", "--format", "%p")]
    [InlineData("--jsonl", "--format", "%l")]
    [InlineData("-s", "--json")]
    [InlineData("-i", "-l")]
    [InlineData("--colors", "--no-colors")]
    [InlineData("-", "-")]
    public async Task ConflictsAndRepeatStdin_Exit2(params string[] args)
    {
        var stderr = new StringWriter();
        int code = await MagikaCommand.RunAsync(args, new StringWriter(), stderr, stdin: null);
        Assert.Equal(2, code);
        Assert.False(string.IsNullOrWhiteSpace(stderr.ToString()));
    }

    [Fact]
    public async Task UnknownOptionAndBadValues_Exit2()
    {
        Assert.Equal(2, await MagikaCommand.RunAsync(["--nope"], new StringWriter(), new StringWriter(), null));
        Assert.Equal(2, await MagikaCommand.RunAsync(["--batch-size", "0"], new StringWriter(), new StringWriter(), null));
        Assert.Equal(2, await MagikaCommand.RunAsync(["--batch-size"], new StringWriter(), new StringWriter(), null));
        Assert.Equal(2, await MagikaCommand.RunAsync(["--workers", "-1"], new StringWriter(), new StringWriter(), null));
        Assert.Equal(2, await MagikaCommand.RunAsync(["--prediction-mode", "nope"], new StringWriter(), new StringWriter(), null));
        Assert.Equal(2, await MagikaCommand.RunAsync(["--format"], new StringWriter(), new StringWriter(), null));
        Assert.Equal(2, await MagikaCommand.RunAsync(["--json=true"], new StringWriter(), new StringWriter(), null));
    }

    [Fact]
    public void SelectFormat_MatchesUpstreamDefaults()
    {
        Assert.Equal("%p: %d (%g)%b", MagikaCommand.SelectFormat(false, false, false, null));
        Assert.Equal("%p: %d (%g)%b %S", MagikaCommand.SelectFormat(true, false, false, null));
        Assert.Equal("%p: %m%b", MagikaCommand.SelectFormat(false, true, false, null));
        Assert.Equal("%p: %m%b %S", MagikaCommand.SelectFormat(true, true, false, null));
        Assert.Equal("%p: %l%b", MagikaCommand.SelectFormat(false, false, true, null));
        Assert.Equal("%p: %l%b %S", MagikaCommand.SelectFormat(true, false, true, null));
        Assert.Equal("%l", MagikaCommand.SelectFormat(true, true, true, "%l"));
    }

    [Fact]
    public void FormatText_PlaceholdersScoresAndOverrule()
    {
        ContentTypeInfo python = ContentTypes.Get("python");
        var kept = new MagikaResult("code.py", python, python, 0.25f, overwriteReason: null);
        Assert.Equal("code.py: Python source (code)", MagikaCommand.FormatText("%p: %d (%g)%b", "code.py", kept, error: null, color: false));
        Assert.Equal("code.py: Python source (code) 25%", MagikaCommand.FormatText("%p: %d (%g)%b %S", "code.py", kept, error: null, color: false));
        Assert.Equal("text/x-python", MagikaCommand.FormatText("%m", "code.py", kept, error: null, color: false));
        Assert.Equal("python", MagikaCommand.FormatText("%l", "code.py", kept, error: null, color: false));
        Assert.Equal("[py,pyi]", MagikaCommand.FormatText("%e", "code.py", kept, error: null, color: false));
        Assert.Equal("0.25", MagikaCommand.FormatText("%s", "code.py", kept, error: null, color: false));
        Assert.Equal("100%", MagikaCommand.FormatText("100%%", "code.py", kept, error: null, color: false));
        Assert.Equal("x", MagikaCommand.FormatText("%x", "code.py", kept, error: null, color: false));
        Assert.Equal("code.py: python 0.25 text/x-python code", MagikaCommand.FormatText("%p: %l %s %m %g", "code.py", kept, error: null, color: false));

        ContentTypeInfo pdf = ContentTypes.Get("pdf");
        ContentTypeInfo unknown = ContentTypes.Get("unknown");
        var low = new MagikaResult("a.bin", unknown, pdf, 0.25f, OverwriteReason.LowConfidence);
        Assert.Equal(
            "a.bin: Unknown binary data (unknown) [Low-confidence model best-guess: PDF document (document), score=0.250]",
            MagikaCommand.FormatText("%p: %d (%g)%b", "a.bin", low, error: null, color: false));

        ContentTypeInfo random = ContentTypes.Get("randombytes");
        var mapped = new MagikaResult("a.bin", unknown, random, 0.9f, OverwriteReason.OverwriteMap);
        Assert.Equal("a.bin: Unknown binary data (unknown)", MagikaCommand.FormatText("%p: %d (%g)%b", "a.bin", mapped, error: null, color: false));

        string colored = MagikaCommand.FormatText("%l", "code.py", kept, error: null, color: true);
        Assert.Contains("\u001b[38;2;139;92;246m", colored);
        Assert.Contains("python", colored);
        Assert.EndsWith("\u001b[0m", colored);

        string errorLine = MagikaCommand.FormatText("%p: %l", "missing", result: null, new FileNotFoundException("gone"), color: true);
        Assert.StartsWith("\u001b[1;31m", errorLine);
        Assert.Contains("missing: error", errorLine);
    }

    [Fact]
    public void FormatDocument_UsesUpstreamSchema()
    {
        var directory = new MagikaResult("dir", ContentTypes.Get("directory"), modelContentType: null, score: 1f, overwriteReason: null);
        string pretty = MagikaCommand.FormatDocument("dir", directory, error: null, pretty: true);
        Assert.Equal(
            """
            {
              "path": "dir",
              "result": {
                "status": "ok",
                "value": {
                  "dl": {
                    "description": "Undefined",
                    "extensions": [],
                    "group": "undefined",
                    "is_text": false,
                    "label": "undefined",
                    "mime_type": "application/undefined"
                  },
                  "output": {
                    "description": "A directory",
                    "extensions": [],
                    "group": "inode",
                    "is_text": false,
                    "label": "directory",
                    "mime_type": "inode/directory"
                  },
                  "score": 1.0
                }
              }
            }
            """.Replace("\r\n", "\n", StringComparison.Ordinal),
            pretty);

        ContentTypeInfo python = ContentTypes.Get("python");
        var kept = new MagikaResult("a\"b\\c", python, python, 0.9969f, overwriteReason: null);
        using JsonDocument compact = JsonDocument.Parse(MagikaCommand.FormatDocument("a\"b\\c", kept, error: null, pretty: false));
        Assert.Equal("a\"b\\c", compact.RootElement.GetProperty("path").GetString());
        Assert.Equal("ok", compact.RootElement.GetProperty("result").GetProperty("status").GetString());
        Assert.Equal("python", compact.RootElement.GetProperty("result").GetProperty("value").GetProperty("dl").GetProperty("label").GetString());
        Assert.Equal("python", compact.RootElement.GetProperty("result").GetProperty("value").GetProperty("output").GetProperty("label").GetString());
        Assert.Equal(0.996, compact.RootElement.GetProperty("result").GetProperty("value").GetProperty("score").GetDouble(), precision: 6);
        Assert.Contains("\"path\":\"a\\\"b\\\\c\"", MagikaCommand.FormatDocument("a\"b\\c", kept, error: null, pretty: false));

        Assert.Equal(
            "{\"path\":\"missing\",\"result\":{\"status\":\"file_does_not_exist\"}}",
            MagikaCommand.FormatDocument("missing", result: null, new FileNotFoundException("gone"), pretty: false));
        Assert.Equal(
            "{\"path\":\"locked\",\"result\":{\"status\":\"permission_error\"}}",
            MagikaCommand.FormatDocument("locked", result: null, new UnauthorizedAccessException("denied"), pretty: false));
        Assert.Contains(
            "\"status\": \"file_does_not_exist\"",
            MagikaCommand.FormatDocument("missing", result: null, new FileNotFoundException("gone"), pretty: true));
    }

    [Fact]
    public void ExpandArgument_GlobsMissingPatternsAndKeepsLiteralNames()
    {
        string root = NewDirectory();
        try
        {
            string sub = Path.Combine(root, "files");
            Directory.CreateDirectory(sub);
            File.WriteAllBytes(Path.Combine(sub, "a.txt"), []);
            File.WriteAllBytes(Path.Combine(sub, "b.bin"), []);
            File.WriteAllBytes(Path.Combine(sub, "a1.txt"), []);
            File.WriteAllBytes(Path.Combine(sub, "a12.txt"), []);

            List<string> stars = MagikaCommand.ExpandArgument(Path.Combine(sub, "*.txt"));
            Assert.Equal(
                [Path.Combine(sub, "a.txt"), Path.Combine(sub, "a1.txt"), Path.Combine(sub, "a12.txt")],
                stars);

            List<string> questions = MagikaCommand.ExpandArgument(Path.Combine(sub, "a?.txt"));
            Assert.Equal([Path.Combine(sub, "a1.txt")], questions);

            string literal = Path.Combine(root, "plain.txt");
            File.WriteAllBytes(literal, []);
            Assert.Equal([literal], MagikaCommand.ExpandArgument(literal));

            if (!OperatingSystem.IsWindows())
            {
                string wildcardName = Path.Combine(sub, "a*b");
                File.WriteAllBytes(wildcardName, []);
                File.WriteAllBytes(Path.Combine(sub, "axb"), []);
                Assert.Equal([wildcardName], MagikaCommand.ExpandArgument(wildcardName));
                File.Delete(wildcardName);
                List<string> expanded = MagikaCommand.ExpandArgument(wildcardName);
                Assert.Equal([Path.Combine(sub, "axb")], expanded);
            }
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task Directory_WithoutRecursive_IdentifiesTheDirectoryItself()
    {
        string root = NewDirectory();
        try
        {
            var stdout = new StringWriter();
            int code = await MagikaCommand.RunAsync([root], stdout, new StringWriter(), stdin: null);
            Assert.Equal(0, code);
            Assert.Equal($"{root}: A directory (inode)\n", stdout.ToString());
            Assert.DoesNotContain("\u001b[", stdout.ToString());

            stdout = new StringWriter();
            code = await MagikaCommand.RunAsync(["--colors", "-s", "-i", root], stdout, new StringWriter(), stdin: null);
            Assert.Equal(0, code);
            string colored = stdout.ToString();
            Assert.Contains("\u001b[", colored);
            Assert.Contains($"{root}: inode/directory 100%", colored);

            stdout = new StringWriter();
            code = await MagikaCommand.RunAsync(["--format", "%l", root], stdout, new StringWriter(), stdin: null);
            Assert.Equal(0, code);
            Assert.Equal("directory\n", stdout.ToString());

            stdout = new StringWriter();
            code = await MagikaCommand.RunAsync(["-r", root], stdout, new StringWriter(), stdin: null);
            Assert.Equal(0, code);
            Assert.Equal("", stdout.ToString());
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task Recursive_ListsFilesInOrderAndSkipsDirectoryInodes()
    {
        string root = NewDirectory();
        try
        {
            File.WriteAllBytes(Path.Combine(root, "z.txt"), []);
            File.WriteAllBytes(Path.Combine(root, "a.txt"), []);
            string sub = Path.Combine(root, "sub");
            Directory.CreateDirectory(sub);
            File.WriteAllBytes(Path.Combine(sub, "c.txt"), []);

            var stdout = new StringWriter();
            int code = await MagikaCommand.RunAsync(["-rl", root], stdout, new StringWriter(), stdin: null);
            Assert.Equal(0, code);
            Assert.Equal(
                string.Join('\n',
                    $"{Path.Combine(root, "a.txt")}: empty",
                    $"{Path.Combine(sub, "c.txt")}: empty",
                    $"{Path.Combine(root, "z.txt")}: empty") + "\n",
                stdout.ToString());
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task Stdin_IsIdentifiedOnce()
    {
        using var stdin = new MemoryStream();
        var stdout = new StringWriter();
        int code = await MagikaCommand.RunAsync(["-l", "-"], stdout, new StringWriter(), stdin);
        Assert.Equal(0, code);
        Assert.Equal("-: empty\n", stdout.ToString());
    }

    [Fact]
    public async Task MissingPath_DoesNotDropLaterResults()
    {
        string root = NewDirectory();
        try
        {
            string kept = Path.Combine(root, "kept.txt");
            File.WriteAllBytes(kept, []);
            string missing = Path.Combine(root, "missing.txt");
            var stdout = new StringWriter();
            int code = await MagikaCommand.RunAsync(["--jsonl", missing, kept, missing], stdout, new StringWriter(), stdin: null);
            Assert.Equal(1, code);
            string[] lines = stdout.ToString().TrimEnd('\n').Split('\n');
            Assert.Equal(3, lines.Length);
            Assert.Contains("\"status\":\"file_does_not_exist\"", lines[0]);
            Assert.Contains("\"status\":\"ok\"", lines[1]);
            Assert.Contains("empty", lines[1]);
            Assert.Contains("\"status\":\"file_does_not_exist\"", lines[2]);
            using JsonDocument ok = JsonDocument.Parse(lines[1]);
            Assert.Equal(kept, ok.RootElement.GetProperty("path").GetString());
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task Json_WrapsResultsInAnArray()
    {
        string root = NewDirectory();
        try
        {
            var stdout = new StringWriter();
            int code = await MagikaCommand.RunAsync(["--json", root], stdout, new StringWriter(), stdin: null);
            Assert.Equal(0, code);
            using JsonDocument document = JsonDocument.Parse(stdout.ToString());
            JsonElement item = document.RootElement[0];
            Assert.Equal(root, item.GetProperty("path").GetString());
            Assert.Equal("directory", item.GetProperty("result").GetProperty("value").GetProperty("output").GetProperty("label").GetString());
            Assert.Equal("undefined", item.GetProperty("result").GetProperty("value").GetProperty("dl").GetProperty("label").GetString());
            Assert.StartsWith("[\n", stdout.ToString());
            Assert.EndsWith("]\n", stdout.ToString());
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task Symlink_FollowsUnlessNoDereference_AndCycleIsPerPath()
    {
        string root = NewDirectory();
        try
        {
            string file = Path.Combine(root, "a.txt");
            File.WriteAllBytes(file, []);
            string link = Path.Combine(root, "link.txt");
            File.CreateSymbolicLink(link, file);

            var stdout = new StringWriter();
            int code = await MagikaCommand.RunAsync(["-l", "--no-dereference", link], stdout, new StringWriter(), stdin: null);
            Assert.Equal(0, code);
            Assert.Equal($"{link}: symlink\n", stdout.ToString());

            stdout = new StringWriter();
            code = await MagikaCommand.RunAsync(["-l", link], stdout, new StringWriter(), stdin: null);
            Assert.Equal(0, code);
            Assert.Equal($"{link}: empty\n", stdout.ToString());

            string loop = Path.Combine(root, "loop");
            try
            {
                Directory.CreateSymbolicLink(loop, root);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return;
            }
            stdout = new StringWriter();
            code = await MagikaCommand.RunAsync(["-r", root], stdout, new StringWriter(), stdin: null);
            Assert.Equal(1, code);
            string text = stdout.ToString();
            Assert.Contains($"{file}: Empty file (inode)", text);
            Assert.Contains($"{loop}: Directory cycle", text);

            stdout = new StringWriter();
            code = await MagikaCommand.RunAsync(["--jsonl", "-r", root], stdout, new StringWriter(), stdin: null);
            Assert.Equal(1, code);
            Assert.Contains("\"status\":\"directory_cycle\"", stdout.ToString());

            stdout = new StringWriter();
            code = await MagikaCommand.RunAsync(["-rl", "--no-dereference", root], stdout, new StringWriter(), stdin: null);
            Assert.Equal(0, code);
            string labels = stdout.ToString();
            Assert.Contains($"{file}: empty", labels);
            Assert.Contains($"{link}: symlink", labels);
            Assert.Contains($"{loop}: symlink", labels);
            Assert.DoesNotContain("error", labels);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task Glob_IsAppliedBeforeIdentification()
    {
        string root = NewDirectory();
        try
        {
            string sub = Path.Combine(root, "files");
            Directory.CreateDirectory(sub);
            File.WriteAllBytes(Path.Combine(sub, "a.txt"), []);
            File.WriteAllBytes(Path.Combine(sub, "b.bin"), []);
            var stdout = new StringWriter();
            int code = await MagikaCommand.RunAsync(["-l", Path.Combine(sub, "*.txt")], stdout, new StringWriter(), stdin: null);
            Assert.Equal(0, code);
            Assert.Equal($"{Path.Combine(sub, "a.txt")}: empty\n", stdout.ToString());
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task UnreadableFile_DoesNotAbortTheRest()
    {
        if (!OperatingSystem.IsLinux() && !OperatingSystem.IsMacOS())
        {
            return;
        }

        string root = NewDirectory();
        string locked = Path.Combine(root, "locked.txt");
        string kept = Path.Combine(root, "kept.txt");
        try
        {
            File.WriteAllBytes(locked, []);
            File.WriteAllBytes(kept, []);
            File.SetUnixFileMode(locked, UnixFileMode.None);
            try
            {
                File.OpenRead(locked).Dispose();
                return;
            }
            catch (UnauthorizedAccessException)
            {
            }
            catch (IOException)
            {
            }

            var stdout = new StringWriter();
            int code = await MagikaCommand.RunAsync(["--jsonl", locked, kept], stdout, new StringWriter(), stdin: null);
            Assert.Equal(1, code);
            string[] lines = stdout.ToString().TrimEnd('\n').Split('\n');
            Assert.Equal(2, lines.Length);
            Assert.Contains("\"status\":\"permission_error\"", lines[0]);
            Assert.Contains("\"status\":\"ok\"", lines[1]);
        }
        finally
        {
            if (File.Exists(locked))
            {
                File.SetUnixFileMode(locked, UnixFileMode.UserRead | UnixFileMode.UserWrite);
            }

            Directory.Delete(root, recursive: true);
        }
    }

    private static string NewDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "magika-cli-" + Guid.NewGuid().ToString("n"));
        Directory.CreateDirectory(path);
        return path;
    }
}
