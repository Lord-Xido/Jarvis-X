using System.Globalization;
using System.Numerics;
using QSol.GraphicsCodec.Core;

namespace QSol.GraphicsCodec.Export;

public static class ObjExporter
{
    public static void Export(Scene3D scene, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var writer = new StreamWriter(path, false);
        writer.WriteLine("# QSOL Graphics Codec decoded scene");
        var vertexOffset = 1;

        foreach (var entity in scene.Entities)
        {
            writer.WriteLine($"o {Sanitize(entity.Name)}");
            writer.WriteLine(FormattableString.Invariant($"# PBR baseColor={entity.Material.BaseColor.X:F4},{entity.Material.BaseColor.Y:F4},{entity.Material.BaseColor.Z:F4},{entity.Material.BaseColor.W:F4} metallic={entity.Material.Metallic:F4} roughness={entity.Material.Roughness:F4}"));

            foreach (var local in entity.Mesh.Vertices)
            {
                var scaled = local * entity.Transform.Scale;
                var rotated = Vector3.Transform(scaled, entity.Transform.Rotation);
                var world = rotated + entity.Transform.Position;
                writer.WriteLine(string.Create(CultureInfo.InvariantCulture, $"v {world.X:R} {world.Y:R} {world.Z:R}"));
            }

            var indices = entity.Mesh.Indices;
            for (var i = 0; i + 2 < indices.Length; i += 3)
            {
                var a = indices[i] + vertexOffset;
                var b = indices[i + 1] + vertexOffset;
                var c = indices[i + 2] + vertexOffset;
                writer.WriteLine($"f {a} {b} {c}");
            }

            vertexOffset += entity.Mesh.Vertices.Length;
        }
    }

    private static string Sanitize(string value) => value.Replace(' ', '_').Replace('\t', '_');
}
