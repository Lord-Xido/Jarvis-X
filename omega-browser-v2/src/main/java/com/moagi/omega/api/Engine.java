package com.moagi.omega.api;

import com.moagi.omega.api.SemanticScene.Action;
import com.moagi.omega.api.SemanticScene.Snapshot;
import com.moagi.omega.api.Surface.Frame;

import java.net.URI;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Flow;

public final class Engine {
    private Engine() {}

    public record Capabilities(
            boolean javascript,
            boolean webGpu,
            boolean nativeSharedSurfaces,
            boolean semanticSnapshots,
            boolean siteIsolation,
            Set<String> schemes
    ) {
        public Capabilities { schemes = Set.copyOf(schemes); }
    }

    public record SessionConfiguration(
            UUID sessionId,
            boolean privateMode,
            int viewportWidth,
            int viewportHeight,
            String profilePartition
    ) {}

    public sealed interface Event permits
            Event.NavigationStarted,
            Event.NavigationCommitted,
            Event.NavigationFailed,
            Event.FrameReady,
            Event.SnapshotReady,
            Event.Status,
            Event.Crashed,
            Event.CapabilityRequested {

        record NavigationStarted(URI uri) implements Event {}
        record NavigationCommitted(URI uri, Origin origin, String title) implements Event {}
        record NavigationFailed(URI uri, String reason) implements Event {}
        record FrameReady(Frame frame) implements Event {}
        record SnapshotReady(Snapshot snapshot) implements Event {}
        record Status(String message) implements Event {}
        record Crashed(String reason, boolean recoverable) implements Event {}
        record CapabilityRequested(Capability capability, Origin origin, String rationale)
                implements Event {}
    }

    public interface BrowserEngine extends AutoCloseable {
        String id();
        String displayName();
        Capabilities capabilities();
        EngineSession createSession(SessionConfiguration configuration);
        boolean healthy();
        @Override void close();
    }

    public interface EngineSession extends AutoCloseable {
        UUID id();
        CompletionStage<Void> navigate(URI uri);
        CompletionStage<Void> reload();
        CompletionStage<Void> stop();
        CompletionStage<Void> execute(long nodeId, Action action, Map<String, String> arguments);
        CompletionStage<Snapshot> snapshot();
        Flow.Publisher<Event> events();
        URI currentUri();
        Origin currentOrigin();
        @Override void close();
    }
}
