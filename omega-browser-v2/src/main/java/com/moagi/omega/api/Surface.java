package com.moagi.omega.api;

import com.moagi.omega.api.Geometry.DamageRect;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Frame delivery abstraction: CPU pixels for the demonstrator or a native shared handle. */
public final class Surface {
    private Surface() {}

    public sealed interface Frame permits CpuFrame, NativeFrame {
        long sequence();
        int width();
        int height();
        List<DamageRect> damage();
    }

    public record CpuFrame(
            long sequence,
            int width,
            int height,
            int[] argb,
            List<DamageRect> damage
    ) implements Frame {
        public CpuFrame {
            if (width <= 0 || height <= 0) throw new IllegalArgumentException("Invalid frame size");
            if (argb.length != width * height) throw new IllegalArgumentException("Pixel array size mismatch");
            argb = Arrays.copyOf(argb, argb.length);
            damage = List.copyOf(damage);
        }

        @Override
        public int[] argb() {
            return Arrays.copyOf(argb, argb.length);
        }
    }

    public record NativeFrame(
            long sequence,
            int width,
            int height,
            long nativeHandle,
            String handleType,
            List<DamageRect> damage
    ) implements Frame {
        public NativeFrame {
            if (width <= 0 || height <= 0) throw new IllegalArgumentException("Invalid frame size");
            handleType = Objects.requireNonNull(handleType, "handleType");
            damage = List.copyOf(damage);
        }
    }
}
