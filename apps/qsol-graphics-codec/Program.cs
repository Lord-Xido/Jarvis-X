using System.Globalization;
using QSol.GraphicsCodec.Codec;
using QSol.GraphicsCodec.Core;
using QSol.GraphicsCodec.Export;
using QSol.GraphicsCodec.Procedural;

namespace QSol.GraphicsCodec;

internal static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            if (args.Contains("--self-test", StringComparer.OrdinalIgnoreCase))
                return SelfTest();

            var decodePath = GetOption(args, "--decode");
            if (decodePath is not null)
            {
                var decoded = SceneCodec.Decode(File.ReadAllBytes(decodePath));
                var objPath = GetOption(args, "--obj") ?? Path.ChangeExtension(decodePath, ".obj");
                ObjExporter.Export(decoded, objPath);
                Console.WriteLine($"Decoded {decoded.Entities.Count} entities -> {objPath}");
                return 0;
            }

            var frames = ParseInt(GetOption(args, "--frames"), 24, 1, 10_000);
            var targetError = ParseFloat(GetOption(args, "--target-error"), 0.0025f, 1e-7f, 1f);
            var output = GetOption(args, "--output") ?? Path.Combine("artifacts", "qsol-graphics-codec");
            Directory.CreateDirectory(output);

            var scene = DemoSceneFactory.Create();
            var metrics = new List<string> { "frame,bits,encoded_bytes,raw_bytes,max_vertex_error" };
            Console.WriteLine($"QSOL Graphics Codec | frames={frames} targetError={targetError:R}");

            for (var frame = 0; frame < frames; frame++)
            {
                var result = SceneCodec.AutoEncode(scene, targetError, compress: true);
                var framePath = Path.Combine(output, $"frame-{frame:D4}.q3d");
                File.WriteAllBytes(framePath, result.Encoded.Data);
                metrics.Add(FormattableString.Invariant($"{frame},{result.Encoded.QuantizationBits},{result.Encoded.Data.Length},{result.Encoded.RawBytes},{result.MaxVertexError:R}"));

                if (frame == 0 || frame == frames - 1)
                    ObjExporter.Export(result.Decoded, Path.Combine(output, $"frame-{frame:D4}.obj"));

                Console.WriteLine(FormattableString.Invariant($"frame={frame:D4} bits={result.Encoded.QuantizationBits} encoded={result.Encoded.Data.Length}B raw={result.Encoded.RawBytes}B maxError={result.MaxVertexError:G6}"));
                KineticIntegrator.Step(scene, 1f / 60f);
            }

            File.WriteAllLines(Path.Combine(output, "metrics.csv"), metrics);
            Console.WriteLine($"Wrote encoded animation stream and decoded OBJ checkpoints to {Path.GetFullPath(output)}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"QSOL Graphics Codec failed: {ex.Message}");
            return 1;
        }
    }

    private static int SelfTest()
    {
        const float tolerance = 0.0025f;
        var scene = DemoSceneFactory.Create();
        var before = scene.Entities[0].Transform.Position;
        KineticIntegrator.Step(scene, 1f / 60f);
        if (scene.Entities[0].Transform.Position == before)
            throw new InvalidOperationException("Kinetic integrator did not advance entity state.");

        var result = SceneCodec.AutoEncode(scene, tolerance, compress: true);
        if (result.Decoded.Entities.Count != scene.Entities.Count)
            throw new InvalidOperationException("Entity count changed during round trip.");
        if (result.MaxVertexError > tolerance)
            throw new InvalidOperationException($"Round-trip error {result.MaxVertexError} exceeds {tolerance}.");

        for (var i = 0; i < scene.Entities.Count; i++)
        {
            if (!scene.Entities[i].Mesh.Indices.SequenceEqual(result.Decoded.Entities[i].Mesh.Indices))
                throw new InvalidOperationException($"Index topology mismatch for entity {i}.");
        }

        var secondDecode = SceneCodec.Decode(result.Encoded.Data);
        if (secondDecode.Entities.Count != scene.Entities.Count)
            throw new InvalidOperationException("Persisted binary stream did not decode deterministically.");

        Console.WriteLine("QSOL Graphics Codec self-test PASS");
        foreach (var candidate in result.Candidates)
        {
            Console.WriteLine(FormattableString.Invariant($"bits={candidate.Bits} encoded={candidate.EncodedBytes}B raw={candidate.RawBytes}B error={candidate.MaxVertexError:G6} meets={candidate.MeetsTolerance}"));
        }
        Console.WriteLine(FormattableString.Invariant($"selected={result.Encoded.QuantizationBits} bits maxError={result.MaxVertexError:G6}"));

        Dmkb1.DmkbSelfTest.Run();
        return 0;
    }

    private static string? GetOption(string[] args, string name)
    {
        for (var i = 0; i < args.Length - 1; i++)
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        return null;
    }

    private static int ParseInt(string? value, int fallback, int min, int max)
    {
        if (value is null) return fallback;
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed) || parsed < min || parsed > max)
            throw new ArgumentException($"Expected integer in range {min}..{max}, got '{value}'.");
        return parsed;
    }

    private static float ParseFloat(string? value, float fallback, float min, float max)
    {
        if (value is null) return fallback;
        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) || parsed < min || parsed > max)
            throw new ArgumentException($"Expected number in range {min}..{max}, got '{value}'.");
        return parsed;
    }
}
