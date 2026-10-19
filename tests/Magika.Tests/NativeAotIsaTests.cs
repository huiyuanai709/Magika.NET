using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace Magika.Tests;

public class NativeAotIsaTests
{
    [Theory]
    [InlineData("linux-x64", "avx2", "avx2", "")]
    [InlineData("linux-x64", "avx512", "avx512", "512")]
    [InlineData("linux-x64", null, "avx2", "")]
    [InlineData("win-x64", "avx2", "avx2", "")]
    [InlineData("win-x64", "avx512", "avx512", "512")]
    public void PublishAot_SelectsInstructionSetAndVectorWidth(string rid, string? isa, string instructionSet, string vectorWidth)
    {
        (string actualSet, string actualWidth) = Evaluate(rid, isa);
        Assert.Equal(instructionSet, actualSet);
        Assert.Equal(vectorWidth, actualWidth);
    }

    [Fact]
    public void PublishAot_OsxArm64SelectsAppleM1()
    {
        (string instructionSet, string vectorWidth) = Evaluate("osx-arm64", isa: null);
        Assert.Equal("apple-m1", instructionSet);
        Assert.Equal("", vectorWidth);
        (string explicitSet, string explicitWidth) = Evaluate("osx-arm64", "apple-m1");
        Assert.Equal("apple-m1", explicitSet);
        Assert.Equal("", explicitWidth);
    }

    [Fact]
    public void PublishAot_RejectsUnknownIsa()
    {
        ProcessResult result = Run(CliProject(), "ValidateMagikaIsa", "linux-x64", "avx512f");
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("MagikaIsa", result.Text, StringComparison.Ordinal);
    }

    [Fact]
    public void PublishAot_RejectsX64IsaOnArm64()
    {
        ProcessResult result = Run(CliProject(), "ValidateMagikaIsa", "osx-arm64", "avx512");
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("apple-m1", result.Text, StringComparison.Ordinal);
    }

    private static (string InstructionSet, string VectorWidth) Evaluate(string rid, string? isa)
    {
        ProcessResult result = Run(CliProject(), target: null, rid, isa);
        Assert.Equal(0, result.ExitCode);
        using JsonDocument doc = JsonDocument.Parse(result.StdOut);
        JsonElement props = doc.RootElement.GetProperty("Properties");
        return (Read(props, "IlcInstructionSet"), Read(props, "IlcMaxVectorTBitWidth"));
    }

    private static string Read(JsonElement props, string name) =>
        props.TryGetProperty(name, out JsonElement value) ? value.GetString() ?? "" : "";

    private static ProcessResult Run(string project, string? target, string rid, string? isa)
    {
        var args = new List<string> { "msbuild", project, "-nologo", "-p:PublishAot=true", $"-p:RuntimeIdentifier={rid}", "-p:EnableWindowsTargeting=true" };
        if (target is null)
        {
            args.Add("-getProperty:IlcInstructionSet");
            args.Add("-getProperty:IlcMaxVectorTBitWidth");
        }
        else
        {
            args.Add($"-t:{target}");
        }

        if (isa is not null)
        {
            args.Add($"-p:MagikaIsa={isa}");
        }

        var start = new ProcessStartInfo("dotnet")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (string arg in args)
        {
            start.ArgumentList.Add(arg);
        }

        using Process process = Process.Start(start) ?? throw new InvalidOperationException("Could not start dotnet.");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        if (!process.WaitForExit(120_000))
        {
            process.Kill(entireProcessTree: true);
            throw new TimeoutException("dotnet msbuild did not finish.");
        }

        return new ProcessResult(process.ExitCode, stdout, stderr);
    }

    private static string CliProject()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string project = Path.Combine(dir.FullName, "src", "Magika.Cli", "Magika.Cli.csproj");
            if (File.Exists(project))
            {
                return project;
            }

            dir = dir.Parent;
        }

        throw new InvalidOperationException("Could not find src/Magika.Cli/Magika.Cli.csproj.");
    }

    private readonly record struct ProcessResult(int ExitCode, string StdOut, string StdErr)
    {
        public string Text => StdOut + StdErr;
    }
}
