using System.Numerics;

namespace QSol.GraphicsCodec.Core;

public readonly record struct Transform3D(Vector3 Position, Quaternion Rotation, Vector3 Scale)
{
    public static Transform3D Identity => new(Vector3.Zero, Quaternion.Identity, Vector3.One);
}

public sealed record MaterialPbr(Vector4 BaseColor, float Metallic, float Roughness);

public sealed class MeshData
{
    public required Vector3[] Vertices { get; init; }
    public required int[] Indices { get; init; }
}

public sealed class Entity3D
{
    public required string Name { get; init; }
    public required MeshData Mesh { get; init; }
    public required MaterialPbr Material { get; init; }
    public Transform3D Transform { get; set; } = Transform3D.Identity;
    public Vector3 LinearVelocity { get; set; }
    public Vector3 AngularVelocity { get; set; }
}

public sealed class Scene3D
{
    public List<Entity3D> Entities { get; } = new();

    public Bounds3D ComputeLocalGeometryBounds()
    {
        var hasPoint = false;
        var min = new Vector3(float.PositiveInfinity);
        var max = new Vector3(float.NegativeInfinity);

        foreach (var entity in Entities)
        {
            foreach (var v in entity.Mesh.Vertices)
            {
                hasPoint = true;
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
        }

        if (!hasPoint)
            return new Bounds3D(Vector3.Zero, Vector3.One);

        var size = max - min;
        if (MathF.Abs(size.X) < 1e-12f) max.X = min.X + 1f;
        if (MathF.Abs(size.Y) < 1e-12f) max.Y = min.Y + 1f;
        if (MathF.Abs(size.Z) < 1e-12f) max.Z = min.Z + 1f;
        return new Bounds3D(min, max);
    }
}

public readonly record struct Bounds3D(Vector3 Min, Vector3 Max)
{
    public Vector3 Size => Max - Min;
}

public static class SceneMetrics
{
    public static float MaxVertexError(Scene3D a, Scene3D b)
    {
        if (a.Entities.Count != b.Entities.Count)
            return float.PositiveInfinity;

        var max = 0f;
        for (var e = 0; e < a.Entities.Count; e++)
        {
            var av = a.Entities[e].Mesh.Vertices;
            var bv = b.Entities[e].Mesh.Vertices;
            if (av.Length != bv.Length)
                return float.PositiveInfinity;

            for (var i = 0; i < av.Length; i++)
                max = MathF.Max(max, Vector3.Distance(av[i], bv[i]));
        }

        return max;
    }
}
