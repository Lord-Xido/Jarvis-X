package com.moagi.omega.adapter.jcef;

import com.moagi.omega.api.Capability;
import com.moagi.omega.api.Engine.CapabilityResolution;
import com.moagi.omega.api.Engine.SessionConfiguration;
import com.moagi.omega.api.Origin;
import com.moagi.omega.api.SemanticScene.Action;
import com.moagi.omega.api.SemanticScene.Snapshot;
import com.moagi.omega.api.Surface.Frame;

import java.net.URI;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Flow;

/**
 * Decoded boundary contract between the trusted Java adapter and an external
 * JCEF/CEF host process. Wire codecs remain outside the engine-neutral API.
 */
public final class JcefHostProtocol {
    public static final int VERSION = 1;

    private JcefHostProtocol() {}

    public record Hello(
            int protocolVersion,
            String engineId,
            String engineVersion,
            boolean nativeSharedSurfaces,
            boolean semanticSnapshots,
            boolean siteIsolation,
            Set<String> schemes
    ) {
        public Hello {
            engineId = Objects.requireNonNull(engineId, "engineId");
            engineVersion = Objects.requireNonNullElse(engineVersion, "unknown");
            schemes = Set.copyOf(Objects.requireNonNullElse(schemes, Set.of()));
            if (protocolVersion <= 0) {
                throw new IllegalArgumentException("Protocol version must be positive");
            }
            if (schemes.isEmpty()) {
                throw new IllegalArgumentException("JCEF host must advertise at least one URI scheme");
            }
        }
    }

    public sealed interface HostEvent permits
            HostEvent.NavigationStarted,
            HostEvent.NavigationCommitted,
            HostEvent.NavigationFailed,
            HostEvent.FrameReady,
            HostEvent.SnapshotReady,
            HostEvent.Status,
            HostEvent.Crashed,
            HostEvent.CapabilityRequested {

        UUID sessionId();

        record NavigationStarted(UUID sessionId, URI uri) implements HostEvent {}

        record NavigationCommitted(
                UUID sessionId,
                URI uri,
                Origin origin,
                String title
        ) implements HostEvent {}

        record NavigationFailed(UUID sessionId, URI uri, String reason) implements HostEvent {}

        record FrameReady(UUID sessionId, Frame frame) implements HostEvent {}

        record SnapshotReady(UUID sessionId, Snapshot snapshot) implements HostEvent {}

        record Status(UUID sessionId, String message) implements HostEvent {}

        record Crashed(UUID sessionId, String reason, boolean recoverable) implements HostEvent {}

        record CapabilityRequested(
                UUID sessionId,
                UUID requestId,
                Capability capability,
                Origin origin,
                String rationale
        ) implements HostEvent {
            public CapabilityRequested {
                Objects.requireNonNull(sessionId, "sessionId");
                Objects.requireNonNull(requestId, "requestId");
                Objects.requireNonNull(capability, "capability");
                Objects.requireNonNull(origin, "origin");
                rationale = Objects.requireNonNullElse(rationale, "");
            }
        }
    }

    /**
     * Authenticated local transport after wire decoding. Implementations may use
     * Unix-domain sockets, Windows named pipes, or another audited local IPC.
     */
    public interface Transport extends AutoCloseable {
        CompletionStage<Hello> hello();

        CompletionStage<Void> createSession(SessionConfiguration configuration);

        CompletionStage<Void> navigate(UUID sessionId, URI uri);

        CompletionStage<Void> reload(UUID sessionId);

        CompletionStage<Void> stop(UUID sessionId);

        CompletionStage<Void> execute(
                UUID sessionId,
                long nodeId,
                Action action,
                Map<String, String> arguments
        );

        CompletionStage<Snapshot> snapshot(UUID sessionId);

        CompletionStage<Void> resolveCapability(
                UUID sessionId,
                CapabilityResolution resolution
        );

        CompletionStage<Void> closeSession(UUID sessionId);

        Flow.Publisher<HostEvent> events();

        boolean healthy();

        @Override
        void close();
    }
}
