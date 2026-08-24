using System.Numerics;
using QSol.GraphicsCodec.Core;

namespace QSol.GraphicsCodec.Procedural;

public static class DemoSceneFactory
{
    public static Scene3D Create()
    {
        var scene = new Scene3D();
        scene.Entities.Add(CreateEntity("kinetic-ring-a", new Vector3(-2.4f, 0f, 0f), new Vector3(0.05f, 0.10f, 0.02f), new Vector3(0.3f, 0.7f, 0.2f), new Vector4(0.82f, 0.22f, 0.12f, 1f), 0.65f, 0.22f));
        scene.Entities.Add(CreateEntity("kinetic-ring-b", new Vector3(2.4f, 0.2f, 0f), new Vector3(-0.04f, 0.02f, 0.06f), new Vector3(0.8f, 0.25f, 0.45f), new Vector4(0.08f, 0.42f, 0.92f, 1f), 0.25f, 0.12f));
        scene.Entities.Add(CreateEntity("kinetic-ring-c", new Vector3(0f, 2.2f, -0.6f), new Vector3(0.02f, -0.05f, 0.03f), new Vector3(0.35f, 0.3f, 0.9f), new Vector4(0.15f, 0.85f, 0.48f, 1f), 0.45f, 0.3f));
        return scene;
    }

    private static Entity3D CreateEntity(string name, Vector3 position, Vector3 velocity, Vector3 angularVelocity, Vector4 color, float metallic, float roughness)
    {
        return new Entity3D
        {
            Name = name,
            Mesh = CreateTorus(48, 18, 1.2f, 0.34f),
            Material = new MaterialPbr(color, metallic, roughness),
            Transform = new Transform3D(position, Quaternion.Identity, Vector3.One),
            LinearVelocity = velocity,
            AngularVelocity = angularVelocity
        };
    }

    public static MeshData CreateTorus(int majorSegments, int minorSegments, float majorRadius, float minorRadius)
    {
        if (majorSegments < 3 || minorSegments < 3) throw new ArgumentOutOfRangeException(nameof(majorSegments));
        var vertices = new Vector3[majorSegments * minorSegments];
        var indices = new int[majorSegments * minorSegments * 6];

        for (var i = 0; i < majorSegments; i++)
        {
            var u = 2f * MathF.PI * i / majorSegments;
            var cu = MathF.Cos(u);
            var su = MathF.Sin(u);
            for (var j = 0; j < minorSegments; j++)
            {
                var v = 2f * MathF.PI * j / minorSegments;
                var cv = MathF.Cos(v);
                var sv = MathF.Sin(v);
                var radius = majorRadius + minorRadius * cv;
                vertices[i * minorSegments + j] = new Vector3(radius * cu, minorRadius * sv, radius * su);
            }
        }

        var k = 0;
        for (var i = 0; i < majorSegments; i++)
        {
            var ni = (i + 1) % majorSegments;
            for (var j = 0; j < minorSegments; j++)
            {
                var nj = (j + 1) % minorSegments;
                var a = i * minorSegments + j;
                var b = ni * minorSegments + j;
                var c = ni * minorSegments + nj;
                var d = i * minorSegments + nj;
                indices[k++] = a; indices[k++] = b; indices[k++] = c;
                indices[k++] = a; indices[k++] = c; indices[k++] = d;
            }
        }

        return new MeshData { Vertices = vertices, Indices = indices };
    }
}
