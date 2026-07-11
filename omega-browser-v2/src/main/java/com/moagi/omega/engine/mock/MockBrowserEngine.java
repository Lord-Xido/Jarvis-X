package com.moagi.omega.engine.mock;

import com.moagi.omega.api.*;
import com.moagi.omega.api.Engine.*;
import com.moagi.omega.api.Geometry.Rect;
import com.moagi.omega.api.Geometry.Transform3D;
import com.moagi.omega.api.SemanticScene.Action;
import com.moagi.omega.api.SemanticScene.Node;
import com.moagi.omega.api.SemanticScene.Snapshot;
import com.moagi.omega.api.Surface.CpuFrame;

import java.net.URI;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Deterministic renderer used to exercise the kernel without native Chromium/Servo binaries.
 * It renders a semantic browser page, not arbitrary internet HTML.
 */
public final class MockBrowserEngine implements BrowserEngine {
    private final ExecutorService tasks = Executors.newVirtualThreadPerTaskExecutor();
    private final AtomicBoolean healthy = new AtomicBoolean(true);

    @Override public String id() { return "mock-semantic"; }
    @Override public String displayName() { return "Deterministic Semantic Engine"; }

    @Override
    public Capabilities capabilities() {
        return new Capabilities(
                false, false, false, true, true,
                Set.of("https", "http", "about", "mock")
        );
    }

    @Override
    public EngineSession createSession(SessionConfiguration configuration) {
        return new Session(configuration);
    }

    @Override public boolean healthy() { return healthy.get(); }

    @Override
    public void close() {
        healthy.set(false);
        tasks.close();
    }

    private final class Session implements EngineSession {
        private final SessionConfiguration configuration;
        private final SubmissionPublisher<Event> events = new SubmissionPublisher<>(tasks, 128);
        private final AtomicLong frameSequence = new AtomicLong();
        private final AtomicLong revision = new AtomicLong();
        private final AtomicBoolean cancelled = new AtomicBoolean();

        private volatile URI current = URI.create("about:blank");
        private volatile Origin origin = Origin.from(current);
        private volatile Snapshot snapshot = buildSnapshot(current, 0, 0);

        private Session(SessionConfiguration configuration) {
            this.configuration = configuration;
        }

        @Override public UUID id() { return configuration.sessionId(); }

        @Override
        public CompletionStage<Void> navigate(URI uri) {
            cancelled.set(false);
            return CompletableFuture.runAsync(() -> {
                events.submit(new Event.NavigationStarted(uri));
                stages("Acquiring resource", "Decoding document", "Computing semantic geometry");
                if (cancelled.get()) throw new CancellationException("Navigation stopped");

                current = uri;
                origin = Origin.from(uri);
                long rev = revision.incrementAndGet();
                long frame = frameSequence.incrementAndGet();
                snapshot = buildSnapshot(uri, rev, frame);
                events.submit(new Event.SnapshotReady(snapshot));
                events.submit(new Event.FrameReady(render(snapshot,
                        configuration.viewportWidth(), configuration.viewportHeight())));
                events.submit(new Event.NavigationCommitted(uri, origin, titleFor(uri)));
                events.submit(new Event.Status("Committed revision " + rev));
            }, tasks);
        }

        @Override public CompletionStage<Void> reload() { return navigate(current); }

        @Override
        public CompletionStage<Void> stop() {
            cancelled.set(true);
            events.submit(new Event.Status("Cancellation requested"));
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<Void> execute(long nodeId, Action action, Map<String, String> arguments) {
            return CompletableFuture.runAsync(() -> {
                Node node = snapshot.node(nodeId);
                if (!node.actions().contains(action)) {
                    throw new IllegalArgumentException("Node does not support " + action);
                }
                events.submit(new Event.Status("Executed " + action + " on " + node.accessibleName()));
                if (action == Action.CLICK && nodeId == 4) {
                    navigate(URI.create("mock://omega/action?source=semantic-node-4")).toCompletableFuture().join();
                } else {
                    long frame = frameSequence.incrementAndGet();
                    snapshot = buildSnapshot(current, revision.incrementAndGet(), frame);
                    events.submit(new Event.SnapshotReady(snapshot));
                    events.submit(new Event.FrameReady(render(snapshot,
                            configuration.viewportWidth(), configuration.viewportHeight())));
                }
            }, tasks);
        }

        @Override public CompletionStage<Snapshot> snapshot() {
            return CompletableFuture.completedFuture(snapshot);
        }

        @Override public Flow.Publisher<Event> events() { return events; }
        @Override public URI currentUri() { return current; }
        @Override public Origin currentOrigin() { return origin; }

        @Override
        public void close() {
            cancelled.set(true);
            events.close();
        }

        private void stages(String... messages) {
            for (String message : messages) {
                if (cancelled.get()) return;
                events.submit(new Event.Status(message));
                try {
                    Thread.sleep(65);
                } catch (InterruptedException ex) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    private static Snapshot buildSnapshot(URI uri, long revision, long frameSequence) {
        String host = uri.getHost() == null ? uri.getScheme() : uri.getHost();
        List<Node> nodes = List.of(
                new Node(0, -1, "document", "Document", new Rect(0, 0, 1200, 760),
                        Transform3D.identity(), Map.of("uri", uri.toString()), Set.of(),
                        List.of(1L, 2L, 3L, 4L, 5L), 0),
                new Node(1, 0, "banner", "Moagi Ω Browser", new Rect(60, 50, 1080, 100),
                        new Transform3D(0, 0, 8, 0, 0, 0, 1, 1, 1),
                        Map.of("engine", "mock-semantic"), Set.of(Action.FOCUS), List.of(), 1),
                new Node(2, 0, "heading", titleFor(uri), new Rect(90, 190, 650, 70),
                        new Transform3D(0, 0, 18, 0, 0.03, -0.01, 1, 1, 1),
                        Map.of("level", "1"), Set.of(Action.COPY), List.of(), 2),
                new Node(3, 0, "article", "Engine-independent semantic scene", new Rect(90, 300, 650, 260),
                        new Transform3D(0, 0, 28, 0.02, 0.04, 0, 1, 1, 1),
                        Map.of("origin", Origin.from(uri).serialize()), Set.of(Action.FOCUS, Action.COPY),
                        List.of(), 3),
                new Node(4, 0, "button", "Commit semantic action", new Rect(805, 300, 300, 86),
                        new Transform3D(0, 0, 52, -0.02, -0.06, 0.01, 1, 1, 1),
                        Map.of("transactional", "true"), Set.of(Action.CLICK, Action.FOCUS), List.of(), 5),
                new Node(5, 0, "complementary", "Runtime telemetry", new Rect(805, 430, 300, 180),
                        new Transform3D(0, 0, 42, 0, -0.05, 0, 1, 1, 1),
                        Map.of("revision", Long.toString(revision), "frame", Long.toString(frameSequence)),
                        Set.of(Action.FOCUS), List.of(), 4)
        );
        return new Snapshot(uri, Origin.from(uri), revision, frameSequence, Instant.now(), nodes);
    }

    private static CpuFrame render(Snapshot snapshot, int width, int height) {
        int[] pixels = new int[width * height];

        // Headless-safe software rasterization: background gradient plus semantic layers.
        for (int y = 0; y < height; y++) {
            double fy = y / (double) Math.max(1, height - 1);
            for (int x = 0; x < width; x++) {
                double fx = x / (double) Math.max(1, width - 1);
                int r = (int) Math.round(12 + 13 * fx + 3 * fy);
                int g = (int) Math.round(18 + 22 * fx + 5 * fy);
                int b = (int) Math.round(28 + 30 * fx + 8 * fy);
                pixels[y * width + x] = argb(255, r, g, b);
            }
        }

        double sx = width / 1200.0;
        double sy = height / 760.0;
        for (Node node : snapshot.nodes()) {
            if (node.id() == 0) continue;
            Rect r = node.bounds();
            int depth = node.layerDepth();
            int offset = depth * 3;
            int x = clamp((int) Math.round((r.x() + offset) * sx), 0, width - 1);
            int y = clamp((int) Math.round((r.y() - offset) * sy), 0, height - 1);
            int w = Math.max(1, (int) Math.round(r.width() * sx));
            int h = Math.max(1, (int) Math.round(r.height() * sy));

            fillRect(pixels, width, height, x + 8, y + 9, w, h, argb(70, 0, 0, 0));
            fillRect(pixels, width, height, x, y, w, h, colorFor(node.role(), depth));
            strokeRect(pixels, width, height, x, y, w, h, argb(210, 170, 215, 255), 2);

            // A small deterministic role marker makes layers visibly distinct without fonts.
            int markerWidth = Math.max(12, Math.min(w - 12, 20 + node.role().length() * 4));
            fillRect(pixels, width, height, x + 10, y + 10, markerWidth, 8,
                    argb(220, 225, 240, 255));
        }

        return new CpuFrame(snapshot.frameSequence(), width, height, pixels,
                Geometry.fullDamage(width, height));
    }

    private static int colorFor(String role, int depth) {
        return switch (role) {
            case "banner" -> argb(238, 30, 92, 145);
            case "heading" -> argb(238, 56, 72, 112);
            case "article" -> argb(238, 31, 103, 95);
            case "button" -> argb(248, 135, 68, 40);
            case "complementary" -> argb(238, 82, 56, 118);
            default -> argb(235, 40 + depth * 8, 55 + depth * 6, 75 + depth * 5);
        };
    }

    private static void fillRect(
            int[] pixels, int width, int height,
            int x, int y, int rectWidth, int rectHeight, int source
    ) {
        int x0 = clamp(x, 0, width);
        int y0 = clamp(y, 0, height);
        int x1 = clamp(x + rectWidth, 0, width);
        int y1 = clamp(y + rectHeight, 0, height);
        for (int py = y0; py < y1; py++) {
            int row = py * width;
            for (int px = x0; px < x1; px++) {
                pixels[row + px] = blend(source, pixels[row + px]);
            }
        }
    }

    private static void strokeRect(
            int[] pixels, int width, int height,
            int x, int y, int rectWidth, int rectHeight, int color, int thickness
    ) {
        fillRect(pixels, width, height, x, y, rectWidth, thickness, color);
        fillRect(pixels, width, height, x, y + rectHeight - thickness, rectWidth, thickness, color);
        fillRect(pixels, width, height, x, y, thickness, rectHeight, color);
        fillRect(pixels, width, height, x + rectWidth - thickness, y, thickness, rectHeight, color);
    }

    private static int blend(int source, int destination) {
        int sa = (source >>> 24) & 0xFF;
        if (sa == 255) return source;
        if (sa == 0) return destination;
        int da = (destination >>> 24) & 0xFF;
        int sr = (source >>> 16) & 0xFF;
        int sg = (source >>> 8) & 0xFF;
        int sb = source & 0xFF;
        int dr = (destination >>> 16) & 0xFF;
        int dg = (destination >>> 8) & 0xFF;
        int db = destination & 0xFF;
        int inv = 255 - sa;
        int oa = sa + (da * inv + 127) / 255;
        int or = (sr * sa + dr * inv + 127) / 255;
        int og = (sg * sa + dg * inv + 127) / 255;
        int ob = (sb * sa + db * inv + 127) / 255;
        return argb(oa, or, og, ob);
    }

    private static int argb(int a, int r, int g, int b) {
        return (clamp(a, 0, 255) << 24)
                | (clamp(r, 0, 255) << 16)
                | (clamp(g, 0, 255) << 8)
                | clamp(b, 0, 255);
    }

    private static int clamp(int value, int minimum, int maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static String titleFor(URI uri) {
        String host = uri.getHost();
        if (host != null && !host.isBlank()) return "Runtime surface: " + host;
        return "Runtime surface: " + uri;
    }
}
