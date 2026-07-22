package com.moagi.omega.api;

import java.util.List;

public final class Geometry {
    private Geometry() {}

    public record Rect(double x, double y, double width, double height) {
        public Rect {
            if (!Double.isFinite(x) || !Double.isFinite(y)
                    || !Double.isFinite(width) || !Double.isFinite(height)
                    || width < 0 || height < 0) {
                throw new IllegalArgumentException("Invalid rectangle");
            }
        }

        public double right() { return x + width; }
        public double bottom() { return y + height; }
        public double centerX() { return x + width / 2.0; }
        public double centerY() { return y + height / 2.0; }
    }

    /** Spatial transform used by the semantic scene and compositor. */
    public record Transform3D(
            double translateX,
            double translateY,
            double translateZ,
            double rotateX,
            double rotateY,
            double rotateZ,
            double scaleX,
            double scaleY,
            double scaleZ
    ) {
        public static Transform3D identity() {
            return new Transform3D(0, 0, 0, 0, 0, 0, 1, 1, 1);
        }
    }

    public record DamageRect(int x, int y, int width, int height) {
        public DamageRect {
            if (x < 0 || y < 0 || width < 0 || height < 0) {
                throw new IllegalArgumentException("Invalid damage rectangle");
            }
        }
    }

    public static List<DamageRect> fullDamage(int width, int height) {
        return List.of(new DamageRect(0, 0, width, height));
    }
}
