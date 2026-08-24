using System.Numerics;

namespace QSol.GraphicsCodec.Core;

public static class KineticIntegrator
{
    public static void Step(Scene3D scene, float dt)
    {
        if (dt <= 0f) return;

        foreach (var entity in scene.Entities)
        {
            var transform = entity.Transform;
            var position = transform.Position + entity.LinearVelocity * dt;
            var rotation = transform.Rotation;

            var speed = entity.AngularVelocity.Length();
            if (speed > 1e-8f)
            {
                var axis = entity.AngularVelocity / speed;
                var delta = Quaternion.CreateFromAxisAngle(axis, speed * dt);
                rotation = Quaternion.Normalize(delta * rotation);
            }

            entity.Transform = new Transform3D(position, rotation, transform.Scale);
        }
    }
}
