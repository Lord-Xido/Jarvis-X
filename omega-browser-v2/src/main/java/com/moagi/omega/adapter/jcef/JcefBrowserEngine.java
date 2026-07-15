package com.moagi.omega.adapter.jcef;

import com.moagi.omega.adapter.jcef.JcefHostProtocol.HostEvent;
import com.moagi.omega.adapter.jcef.JcefHostProtocol.Transport;
import com.moagi.omega.api.Engine;
import com.moagi.omega.api.Engine.BrowserEngine;
import com.moagi.omega.api.Engine.Capabilities;
import com.moagi.omega.api.Engine.CapabilityResolution;
import com.moagi.omega.api.Engine.EngineSession;
import com.moagi.omega.api.Engine.Event;
import com.moagi.omega.api.Engine.SessionConfiguration;
import com.moagi.omega.api.Origin;
import com.moagi.omega.api.SemanticScene.Action;
import com.moagi.omega.api.SemanticScene.Snapshot;
import com.moagi.omega.core.ProcessSupervisor;

import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Flow;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Engine-neutral adapter for a separately packaged JCEF/CEF host process.
 * Chromium classes and renderer execution never enter the trusted kernel JVM.
 */
public final class JcefBrowserEngine implements BrowserEngine {
    @FunctionalInterface
    public interface HostConnector {
        Transport connect(ProcessSupervisor.ProcessState processState) throws Exception;
    }

    public record LaunchConfiguration(
            ProcessSupervisor.ProcessSpec processSpec,
            Duration handshakeTimeout,
            Duration commandTimeout
    ) {
        public LaunchConfiguration {
            Objects.requireNonNull(processSpec, "processSpec");
            handshakeTimeout = positive(handshakeTimeout, "handshakeTimeout");
            commandTimeout = positive(commandTimeout, "commandTimeout");
        }
    }

    private final Transport transport;
    private final Runnable shutdownHook;
    private final Duration commandTimeout;
    private final ExecutorService eventsExecutor = Executors.newVirtualThreadPerTaskExecutor();
    private final Map<UUID, Session> sessions = new ConcurrentHashMap<>();
    private final AtomicBoolean hostHealthy = new AtomicBoolean(true);
    private final AtomicBoolean closed = new AtomicBoolean();
    private final String engineId;
    private final String displayName;
    private final Capabilities capabilities;

    private JcefBrowserEngine(
            Transport transport,
            Runnable shutdownHook,
            Duration handshakeTimeout,
            Duration commandTimeout
    ) {
        this.transport = Objects.requireNonNull(transport, "transport");
        this.shutdownHook = Objects.requireNonNull(shutdownHook, "shutdownHook");
        this.commandTimeout = positive(commandTimeout, "commandTimeout");

        JcefHostProtocol.Hello hello = await(
                transport.hello(),
                positive(handshakeTimeout, "handshakeTimeout"),
                "JCEF host protocol negotiation"
        );
        if (hello.protocolVersion() != JcefHostProtocol.VERSION) {
            throw new IllegalStateException(
                    "Unsupported JCEF host protocol " + hello.protocolVersion()
                            + "; expected " + JcefHostProtocol.VERSION
            );
        }

        this.engineId = "jcef-" + hello.engineId();
        this.displayName = "JCEF/CEF " + hello.engineVersion();
        this.capabilities = new Capabilities(
                true,
                false,
                hello.nativeSharedSurfaces(),
                hello.semanticSnapshots(),
                hello.siteIsolation(),
                hello.schemes()
        );
        subscribeToHost();
    }

    /**
     * Launches and supervises an external JCEF host, then negotiates its local
     * authenticated transport. The caller owns the supervisor lifecycle.
     */
    public static JcefBrowserEngine launch(
            ProcessSupervisor supervisor,
            LaunchConfiguration configuration,
            HostConnector connector
    ) throws IOException {
        Objects.requireNonNull(supervisor, "supervisor");
        Objects.requireNonNull(configuration, "configuration");
        Objects.requireNonNull(connector, "connector");

        ProcessSupervisor.ProcessSpec spec = configuration.processSpec();
        supervisor.start(spec);
        Transport transport = null;
        try {
            ProcessSupervisor.ProcessState state = supervisor.state(spec.id());
            if (state == null || !state.alive()) {
                throw new IllegalStateException("JCEF host did not remain alive after launch");
            }
            transport = connector.connect(state);
            Transport negotiatedTransport = transport;
            return new JcefBrowserEngine(
                    negotiatedTransport,
                    () -> supervisor.stop(spec.id()),
                    configuration.handshakeTimeout(),
                    configuration.commandTimeout()
            );
        } catch (Exception error) {
            if (transport != null) {
                try {
                    transport.close();
                } catch (RuntimeException ignored) {
                    // Preserve the launch failure as the primary exception.
                }
            }
            supervisor.stop(spec.id());
            if (error instanceof RuntimeException runtime) throw runtime;
            throw new IllegalStateException("Cannot connect to supervised JCEF host", error);
        }
    }

    /** Creates an adapter over an already authenticated transport, primarily for tests. */
    public static JcefBrowserEngine connect(
            Transport transport,
            Duration handshakeTimeout,
            Duration commandTimeout
    ) {
        return new JcefBrowserEngine(
                transport,
                () -> {},
                handshakeTimeout,
                commandTimeout
        );
    }

    @Override
    public String id() {
        return engineId;
    }

    @Override
    public String displayName() {
        return displayName;
    }

    @Override
    public Capabilities capabilities() {
        return capabilities;
    }

    @Override
    public EngineSession createSession(SessionConfiguration configuration) {
        Objects.requireNonNull(configuration, "configuration");
        ensureHealthy();

        Session session = new Session(configuration);
        if (sessions.putIfAbsent(configuration.sessionId(), session) != null) {
            throw new IllegalStateException("Duplicate JCEF session: " + configuration.sessionId());
        }
        try {
            await(
                    transport.createSession(configuration),
                    commandTimeout,
                    "create JCEF session " + configuration.sessionId()
            );
            return session;
        } catch (RuntimeException error) {
            sessions.remove(configuration.sessionId(), session);
            session.closePublisher();
            throw error;
        }
    }

    @Override
    public boolean healthy() {
        return !closed.get() && hostHealthy.get() && transport.healthy();
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        for (Session session : sessions.values()) {
            session.close();
        }
        sessions.clear();
        try {
            transport.close();
        } finally {
            shutdownHook.run();
            eventsExecutor.close();
        }
    }

    private void subscribeToHost() {
        transport.events().subscribe(new Flow.Subscriber<>() {
            private Flow.Subscription subscription;

            @Override
            public void onSubscribe(Flow.Subscription subscription) {
                this.subscription = subscription;
                subscription.request(Long.MAX_VALUE);
            }

            @Override
            public void onNext(HostEvent item) {
                route(item);
            }

            @Override
            public void onError(Throwable throwable) {
                hostHealthy.set(false);
                broadcastCrash("JCEF host event channel failed: " + throwable, true);
            }

            @Override
            public void onComplete() {
                if (!closed.get()) {
                    hostHealthy.set(false);
                    broadcastCrash("JCEF host event channel closed", true);
                }
            }
        });
    }

    private void route(HostEvent hostEvent) {
        Session session = sessions.get(hostEvent.sessionId());
        if (session == null || session.closed.get()) return;

        Event event;
        if (hostEvent instanceof HostEvent.NavigationStarted started) {
            event = new Event.NavigationStarted(started.uri());
        } else if (hostEvent instanceof HostEvent.NavigationCommitted committed) {
            session.currentUri = committed.uri();
            session.currentOrigin = committed.origin();
            event = new Event.NavigationCommitted(
                    committed.uri(), committed.origin(), committed.title()
            );
        } else if (hostEvent instanceof HostEvent.NavigationFailed failed) {
            event = new Event.NavigationFailed(failed.uri(), failed.reason());
        } else if (hostEvent instanceof HostEvent.FrameReady frame) {
            event = new Event.FrameReady(frame.frame());
        } else if (hostEvent instanceof HostEvent.SnapshotReady ready) {
            session.latestSnapshot = ready.snapshot();
            event = new Event.SnapshotReady(ready.snapshot());
        } else if (hostEvent instanceof HostEvent.Status status) {
            event = new Event.Status(status.message());
        } else if (hostEvent instanceof HostEvent.Crashed crash) {
            event = new Event.Crashed(crash.reason(), crash.recoverable());
        } else if (hostEvent instanceof HostEvent.CapabilityRequested request) {
            event = new Event.CapabilityRequested(
                    request.requestId(),
                    request.capability(),
                    request.origin(),
                    request.rationale()
            );
        } else {
            event = new Event.Status("Ignored unknown JCEF host event: " + hostEvent);
        }
        session.events.submit(event);
    }

    private void broadcastCrash(String reason, boolean recoverable) {
        for (Session session : sessions.values()) {
            if (!session.closed.get()) {
                session.events.submit(new Event.Crashed(reason, recoverable));
            }
        }
    }

    private void ensureHealthy() {
        if (!healthy()) {
            throw new IllegalStateException("JCEF engine host is not healthy");
        }
    }

    private <T> CompletionStage<T> guarded(CompletionStage<T> operation) {
        if (closed.get()) {
            return CompletableFuture.failedFuture(new IllegalStateException("JCEF engine is closed"));
        }
        return operation.toCompletableFuture()
                .orTimeout(commandTimeout.toMillis(), TimeUnit.MILLISECONDS);
    }

    private final class Session implements EngineSession {
        private final SessionConfiguration configuration;
        private final SubmissionPublisher<Event> events =
                new SubmissionPublisher<>(eventsExecutor, 256);
        private final AtomicBoolean closed = new AtomicBoolean();
        private volatile URI currentUri = URI.create("about:blank");
        private volatile Origin currentOrigin = Origin.from(currentUri);
        private volatile Snapshot latestSnapshot;

        private Session(SessionConfiguration configuration) {
            this.configuration = configuration;
        }

        @Override
        public UUID id() {
            return configuration.sessionId();
        }

        @Override
        public CompletionStage<Void> navigate(URI uri) {
            Objects.requireNonNull(uri, "uri");
            return invoke(() -> transport.navigate(id(), uri));
        }

        @Override
        public CompletionStage<Void> reload() {
            return invoke(() -> transport.reload(id()));
        }

        @Override
        public CompletionStage<Void> stop() {
            return invoke(() -> transport.stop(id()));
        }

        @Override
        public CompletionStage<Void> execute(
                long nodeId,
                Action action,
                Map<String, String> arguments
        ) {
            Objects.requireNonNull(action, "action");
            Map<String, String> safeArguments = Map.copyOf(
                    Objects.requireNonNullElse(arguments, Map.of())
            );
            return invoke(() -> transport.execute(id(), nodeId, action, safeArguments));
        }

        @Override
        public CompletionStage<Void> resolveCapability(CapabilityResolution resolution) {
            Objects.requireNonNull(resolution, "resolution");
            return invoke(() -> transport.resolveCapability(id(), resolution));
        }

        @Override
        public CompletionStage<Snapshot> snapshot() {
            if (closed.get()) {
                return CompletableFuture.failedFuture(
                        new IllegalStateException("JCEF session is closed: " + id())
                );
            }
            return guarded(transport.snapshot(id())).thenApply(snapshot -> {
                latestSnapshot = snapshot;
                return snapshot;
            });
        }

        @Override
        public Flow.Publisher<Event> events() {
            return events;
        }

        @Override
        public URI currentUri() {
            return currentUri;
        }

        @Override
        public Origin currentOrigin() {
            return currentOrigin;
        }

        @Override
        public void close() {
            if (!closed.compareAndSet(false, true)) return;
            sessions.remove(id(), this);
            try {
                await(transport.closeSession(id()), commandTimeout, "close JCEF session " + id());
            } catch (RuntimeException ignored) {
                // Session close remains idempotent even after renderer failure.
            } finally {
                closePublisher();
            }
        }

        private CompletionStage<Void> invoke(
                java.util.function.Supplier<CompletionStage<Void>> operation
        ) {
            if (closed.get()) {
                return CompletableFuture.failedFuture(
                        new IllegalStateException("JCEF session is closed: " + id())
                );
            }
            ensureHealthy();
            try {
                return guarded(operation.get());
            } catch (RuntimeException error) {
                return CompletableFuture.failedFuture(error);
            }
        }

        private void closePublisher() {
            events.close();
        }
    }

    private static Duration positive(Duration value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isZero() || value.isNegative()) {
            throw new IllegalArgumentException(name + " must be positive");
        }
        return value;
    }

    private static <T> T await(
            CompletionStage<T> operation,
            Duration timeout,
            String description
    ) {
        try {
            return operation.toCompletableFuture().get(
                    timeout.toMillis(),
                    TimeUnit.MILLISECONDS
            );
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(description + " interrupted", error);
        } catch (ExecutionException error) {
            Throwable cause = error.getCause() == null ? error : error.getCause();
            throw new IllegalStateException(description + " failed", cause);
        } catch (TimeoutException error) {
            throw new IllegalStateException(description + " timed out", error);
        }
    }
}
