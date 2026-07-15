package com.moagi.omega.api;

import com.moagi.omega.api.SemanticScene.Action;
import com.moagi.omega.api.SemanticScene.Snapshot;
import com.moagi.omega.api.Surface.Frame;

import java.net.URI;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;
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

    /**
     * Kernel-issued identity for one engine operation. The transaction id is
     * stable across the kernel, adapter, IPC transport and native host.
     */
    public record OperationContext(
            UUID transactionId,
            long parentRevision,
            Instant deadline
    ) {
        public OperationContext {
            Objects.requireNonNull(transactionId, "transactionId");
            Objects.requireNonNull(deadline, "deadline");
            if (parentRevision < -1) {
                throw new IllegalArgumentException("parentRevision must be -1 or non-negative");
            }
        }

        public static OperationContext detached() {
            return new OperationContext(UUID.randomUUID(), -1, Instant.now().plusSeconds(30));
        }

        public boolean expired() {
            return !deadline.isAfter(Instant.now());
        }
    }

    /**
     * Kernel-issued response to an engine capability request. A granted response
     * always carries the broker token id used to authorize exactly this request.
     */
    public record CapabilityResolution(
            UUID requestId,
            Capability capability,
            Origin origin,
            boolean granted,
            UUID tokenId,
            String detail
    ) {
        public CapabilityResolution {
            Objects.requireNonNull(requestId, "requestId");
            Objects.requireNonNull(capability, "capability");
            Objects.requireNonNull(origin, "origin");
            detail = Objects.requireNonNullElse(detail, "");
            if (granted && tokenId == null) {
                throw new IllegalArgumentException("Granted capability resolution requires a token id");
            }
            if (!granted && tokenId != null) {
                throw new IllegalArgumentException("Denied capability resolution cannot carry a token id");
            }
        }
    }

    public sealed interface Event permits
            Event.NavigationStarted,
            Event.NavigationCommitted,
            Event.NavigationFailed,
            Event.FrameReady,
            Event.SnapshotReady,
            Event.Status,
            Event.Crashed,
            Event.CapabilityRequested,
            Event.Correlated {

        record NavigationStarted(URI uri) implements Event {}
        record NavigationCommitted(URI uri, Origin origin, String title) implements Event {}
        record NavigationFailed(URI uri, String reason) implements Event {}
        record FrameReady(Frame frame) implements Event {}
        record SnapshotReady(Snapshot snapshot) implements Event {}
        record Status(String message) implements Event {}
        record Crashed(String reason, boolean recoverable) implements Event {}

        /**
         * Correlates an engine callback to the kernel transaction that caused it.
         * documentRevision is -1 when no committed document revision exists yet.
         */
        record Correlated(
                UUID transactionId,
                long documentRevision,
                Event event
        ) implements Event {
            public Correlated {
                Objects.requireNonNull(transactionId, "transactionId");
                Objects.requireNonNull(event, "event");
                if (event instanceof Correlated) {
                    throw new IllegalArgumentException("Correlated events cannot be nested");
                }
                if (documentRevision < -1) {
                    throw new IllegalArgumentException("documentRevision must be -1 or non-negative");
                }
            }
        }

        record CapabilityRequested(
                UUID requestId,
                Capability capability,
                Origin origin,
                String rationale
        ) implements Event {
            public CapabilityRequested {
                Objects.requireNonNull(requestId, "requestId");
                Objects.requireNonNull(capability, "capability");
                Objects.requireNonNull(origin, "origin");
                rationale = Objects.requireNonNullElse(rationale, "");
            }
        }
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

        default CompletionStage<Void> navigate(OperationContext context, URI uri) {
            Objects.requireNonNull(context, "context");
            return navigate(uri);
        }

        default CompletionStage<Void> reload(OperationContext context) {
            Objects.requireNonNull(context, "context");
            return reload();
        }

        default CompletionStage<Void> stop(OperationContext context) {
            Objects.requireNonNull(context, "context");
            return stop();
        }

        default CompletionStage<Void> execute(
                OperationContext context,
                long nodeId,
                Action action,
                Map<String, String> arguments
        ) {
            Objects.requireNonNull(context, "context");
            return execute(nodeId, action, arguments);
        }

        CompletionStage<Void> resolveCapability(CapabilityResolution resolution);
        CompletionStage<Snapshot> snapshot();
        Flow.Publisher<Event> events();
        URI currentUri();
        Origin currentOrigin();
        @Override void close();
    }
}
