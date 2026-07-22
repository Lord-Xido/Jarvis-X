package com.moagi.omega.adapter.jcef;

import com.moagi.omega.adapter.jcef.JcefHostProtocol.HostEvent;
import com.moagi.omega.adapter.jcef.JcefHostProtocol.SessionRecovery;
import com.moagi.omega.adapter.jcef.JcefHostProtocol.Transport;
import com.moagi.omega.api.Engine;
import com.moagi.omega.api.Engine.BrowserEngine;
import com.moagi.omega.api.Engine.Capabilities;
import com.moagi.omega.api.Engine.CapabilityResolution;
import com.moagi.omega.api.Engine.EngineSession;
import com.moagi.omega.api.Engine.Event;
import com.moagi.omega.api.Engine.OperationContext;
import com.moagi.omega.api.Engine.SessionConfiguration;
import com.moagi.omega.api.Origin;
import com.moagi.omega.api.SemanticScene.Action;
import com.moagi.omega.api.SemanticScene.Snapshot;
import com.moagi.omega.core.ProcessSupervisor;

import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.time.Instant;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

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

    private final AtomicReference<Transport> transport;
    private final Runnable shutdownHook;
    private final Duration handshakeTimeout;
    private final Duration commandTimeout;
    private final ExecutorService eventsExecutor = Executors.newVirtualThreadPerTaskExecutor();
    private final Map<UUID, Session> sessions = new ConcurrentHashMap<>();
    private final AtomicBoolean hostHealthy = new AtomicBoolean(true);
    private final AtomicBoolean closed = new AtomicBoolean();
    private final AtomicLong transportGeneration = new AtomicLong(1);
    private final AtomicInteger connectedRestarts = new AtomicInteger();
    private final AtomicReference<AutoCloseable> lifecycleRegistration = new AtomicReference<>();
    private final Object recoveryLock = new Object();
    private final String engineId;
    private final String hostEngineId;
    private final String displayName;
    private final Capabilities capabilities;

    private JcefBrowserEngine(
            Transport initialTransport,
            Runnable shutdownHook,
            Duration handshakeTimeout,
            Duration commandTimeout
    ) {
        this.transport = new AtomicReference<>(Objects.requireNonNull(initialTransport, "initialTransport"));
        this.shutdownHook = Objects.requireNonNull(shutdownHook, "shutdownHook");
        this.handshakeTimeout = positive(handshakeTimeout, "handshakeTimeout");
        this.commandTimeout = positive(commandTimeout, "commandTimeout");

        JcefHostProtocol.Hello hello = negotiate(initialTransport);
        this.hostEngineId = hello.engineId();
        this.engineId = "jcef-" + hostEngineId;
        this.displayName = "JCEF/CEF " + hello.engineVersion();
        this.capabilities = capabilities(hello);
        subscribeToHost(initialTransport, transportGeneration.get());
    }

    /**
     * Launches and supervises an external JCEF host, negotiates its local
     * authenticated transport, and rebinds the adapter after bounded restarts.
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
        Transport initialTransport = null;
        try {
            ProcessSupervisor.ProcessState state = supervisor.state(spec.id());
            if (state == null || !state.alive()) {
                throw new IllegalStateException("JCEF host did not remain alive after launch");
            }
            initialTransport = connector.connect(state);
            JcefBrowserEngine engine = new JcefBrowserEngine(
                    initialTransport,
                    () -> supervisor.stop(spec.id()),
                    configuration.handshakeTimeout(),
                    configuration.commandTimeout()
            );
            AutoCloseable registration = supervisor.subscribe(
                    spec.id(),
                    processState -> engine.onProcessState(processState, connector)
            );
            engine.lifecycleRegistration.set(registration);
            return engine;
        } catch (Exception error) {
            if (initialTransport != null) closeQuietly(initialTransport);
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

    /**
     * Rebinds the adapter to a replacement host and rehydrates every live
     * session before the engine is marked healthy again.
     */
    public void recover(Transport replacement) {
        Objects.requireNonNull(replacement, "replacement");
        synchronized (recoveryLock) {
            if (closed.get()) {
                closeQuietly(replacement);
                throw new IllegalStateException("JCEF engine is closed");
            }

            JcefHostProtocol.Hello hello = negotiate(replacement);
            verifyCompatible(hello);

            Transport previous = transport.get();
            long replacementGeneration = transportGeneration.incrementAndGet();
            transport.set(replacement);
            subscribeToHost(replacement, replacementGeneration);

            try {
                for (Session session : sessions.values()) {
                    if (!session.closed.get()) {
                        await(
                                replacement.recoverSession(session.recovery()),
                                commandTimeout,
                                "recover JCEF session " + session.id()
                        );
                    }
                }
                hostHealthy.set(true);
                closeQuietly(previous);
                for (Session session : sessions.values()) {
                    if (!session.closed.get()) {
                        session.events.submit(new Event.Status("JCEF host recovered"));
                    }
                }
            } catch (RuntimeException error) {
                transport.compareAndSet(replacement, previous);
                transportGeneration.incrementAndGet();
                closeQuietly(replacement);
                hostHealthy.set(false);
                broadcastCrash("JCEF host recovery failed: " + error, false);
                throw error;
            }
        }
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
                    activeTransport().createSession(configuration),
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
        Transport active = transport.get();
        return !closed.get() && hostHealthy.get() && active != null && active.healthy();
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        for (Session session : sessions.values()) {
            session.close();
        }
        sessions.clear();
        try {
            closeQuietly(transport.getAndSet(null));
        } finally {
            AutoCloseable registration = lifecycleRegistration.getAndSet(null);
            if (registration != null) {
                try {
                    registration.close();
                } catch (Exception ignored) {
                    // Shutdown continues even if lifecycle listener removal fails.
                }
            }
            shutdownHook.run();
            eventsExecutor.close();
        }
    }

    private void onProcessState(ProcessSupervisor.ProcessState state, HostConnector connector) {
        if (closed.get()) return;
        if (!state.alive()) {
            if (hostHealthy.getAndSet(false)) {
                broadcastCrash(
                        "JCEF host exited"
                                + (state.exitCode() == null ? "" : " with code " + state.exitCode()),
                        state.restarts() < Integer.MAX_VALUE
                );
            }
            return;
        }
        if (state.restarts() <= connectedRestarts.get()) return;

        eventsExecutor.submit(() -> {
            synchronized (recoveryLock) {
                if (closed.get() || state.restarts() <= connectedRestarts.get()) return;
                try {
                    recover(connector.connect(state));
                    connectedRestarts.set(state.restarts());
                } catch (Exception error) {
                    hostHealthy.set(false);
                    broadcastCrash("Cannot reconnect to restarted JCEF host: " + error, false);
                }
            }
        });
    }

    private void subscribeToHost(Transport subscribedTransport, long generation) {
        subscribedTransport.events().subscribe(new Flow.Subscriber<>() {
            @Override
            public void onSubscribe(Flow.Subscription subscription) {
                subscription.request(Long.MAX_VALUE);
            }

            @Override
            public void onNext(HostEvent item) {
                if (isCurrent(subscribedTransport, generation)) route(item);
            }

            @Override
            public void onError(Throwable throwable) {
                if (isCurrent(subscribedTransport, generation)
                        && hostHealthy.getAndSet(false)) {
                    broadcastCrash("JCEF host event channel failed: " + throwable, true);
                }
            }

            @Override
            public void onComplete() {
                if (!closed.get()
                        && isCurrent(subscribedTransport, generation)
                        && hostHealthy.getAndSet(false)) {
                    broadcastCrash("JCEF host event channel closed", true);
                }
            }
        });
    }

    private boolean isCurrent(Transport subscribedTransport, long generation) {
        return transport.get() == subscribedTransport && transportGeneration.get() == generation;
    }

    private void route(HostEvent hostEvent) {
        Session session = sessions.get(hostEvent.sessionId());
        if (session == null || session.closed.get()) return;

        Event event;
        if (hostEvent instanceof HostEvent.NavigationStarted started) {
            event = correlated(
                    started.transactionId(),
                    -1,
                    new Event.NavigationStarted(started.uri())
            );
        } else if (hostEvent instanceof HostEvent.NavigationCommitted committed) {
            session.currentUri = committed.uri();
            session.currentOrigin = committed.origin();
            session.documentRevision.accumulateAndGet(committed.documentRevision(), Math::max);
            event = correlated(
                    committed.transactionId(),
                    committed.documentRevision(),
                    new Event.NavigationCommitted(
                            committed.uri(), committed.origin(), committed.title()
                    )
            );
        } else if (hostEvent instanceof HostEvent.NavigationFailed failed) {
            event = correlated(
                    failed.transactionId(),
                    -1,
                    new Event.NavigationFailed(failed.uri(), failed.reason())
            );
        } else if (hostEvent instanceof HostEvent.FrameReady frame) {
            session.documentRevision.accumulateAndGet(frame.documentRevision(), Math::max);
            event = correlated(
                    frame.transactionId(),
                    frame.documentRevision(),
                    new Event.FrameReady(frame.frame())
            );
        } else if (hostEvent instanceof HostEvent.SnapshotReady ready) {
            session.latestSnapshot = ready.snapshot();
            session.documentRevision.accumulateAndGet(ready.snapshot().revision(), Math::max);
            event = correlated(
                    ready.transactionId(),
                    ready.snapshot().revision(),
                    new Event.SnapshotReady(ready.snapshot())
            );
        } else if (hostEvent instanceof HostEvent.Status status) {
            event = correlated(
                    status.transactionId(),
                    session.documentRevision.get(),
                    new Event.Status(status.message())
            );
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

    private static Event correlated(UUID transactionId, long revision, Event event) {
        return new Event.Correlated(
                Objects.requireNonNull(transactionId, "transactionId"),
                revision,
                event
        );
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

    private Transport activeTransport() {
        Transport active = transport.get();
        if (active == null) throw new IllegalStateException("JCEF engine transport is closed");
        return active;
    }

    private <T> CompletionStage<T> guarded(
            OperationContext context,
            CompletionStage<T> operation
    ) {
        if (closed.get()) {
            return CompletableFuture.failedFuture(new IllegalStateException("JCEF engine is closed"));
        }
        long remainingMillis = Duration.between(Instant.now(), context.deadline()).toMillis();
        if (remainingMillis <= 0) {
            return CompletableFuture.failedFuture(
                    new TimeoutException("JCEF operation deadline expired: " + context.transactionId())
            );
        }
        long timeoutMillis = Math.min(commandTimeout.toMillis(), remainingMillis);
        return operation.toCompletableFuture().orTimeout(timeoutMillis, TimeUnit.MILLISECONDS);
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
        private final AtomicLong documentRevision = new AtomicLong();
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
            return navigate(OperationContext.detached(), uri);
        }

        @Override
        public CompletionStage<Void> navigate(OperationContext context, URI uri) {
            Objects.requireNonNull(context, "context");
            Objects.requireNonNull(uri, "uri");
            return invoke(context, active -> active.navigate(id(), context, uri));
        }

        @Override
        public CompletionStage<Void> reload() {
            return reload(OperationContext.detached());
        }

        @Override
        public CompletionStage<Void> reload(OperationContext context) {
            Objects.requireNonNull(context, "context");
            return invoke(context, active -> active.reload(id(), context));
        }

        @Override
        public CompletionStage<Void> stop() {
            return stop(OperationContext.detached());
        }

        @Override
        public CompletionStage<Void> stop(OperationContext context) {
            Objects.requireNonNull(context, "context");
            return invoke(context, active -> active.stop(id(), context));
        }

        @Override
        public CompletionStage<Void> execute(
                long nodeId,
                Action action,
                Map<String, String> arguments
        ) {
            return execute(OperationContext.detached(), nodeId, action, arguments);
        }

        @Override
        public CompletionStage<Void> execute(
                OperationContext context,
                long nodeId,
                Action action,
                Map<String, String> arguments
        ) {
            Objects.requireNonNull(context, "context");
            Objects.requireNonNull(action, "action");
            Map<String, String> safeArguments = Map.copyOf(
                    Objects.requireNonNullElse(arguments, Map.of())
            );
            return invoke(
                    context,
                    active -> active.execute(id(), context, nodeId, action, safeArguments)
            );
        }

        @Override
        public CompletionStage<Void> resolveCapability(CapabilityResolution resolution) {
            Objects.requireNonNull(resolution, "resolution");
            if (closed.get()) {
                return CompletableFuture.failedFuture(
                        new IllegalStateException("JCEF session is closed: " + id())
                );
            }
            ensureHealthy();
            try {
                return guarded(activeTransport().resolveCapability(id(), resolution));
            } catch (RuntimeException error) {
                return CompletableFuture.failedFuture(error);
            }
        }

        @Override
        public CompletionStage<Snapshot> snapshot() {
            if (closed.get()) {
                return CompletableFuture.failedFuture(
                        new IllegalStateException("JCEF session is closed: " + id())
                );
            }
            ensureHealthy();
            return guarded(activeTransport().snapshot(id())).thenApply(snapshot -> {
                latestSnapshot = snapshot;
                documentRevision.accumulateAndGet(snapshot.revision(), Math::max);
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
                Transport active = transport.get();
                if (active != null && active.healthy()) {
                    await(active.closeSession(id()), commandTimeout, "close JCEF session " + id());
                }
            } catch (RuntimeException ignored) {
                // Session close remains idempotent even after renderer failure.
            } finally {
                closePublisher();
            }
        }

        private CompletionStage<Void> invoke(
                OperationContext context,
                java.util.function.Function<Transport, CompletionStage<Void>> operation
        ) {
            if (closed.get()) {
                return CompletableFuture.failedFuture(
                        new IllegalStateException("JCEF session is closed: " + id())
                );
            }
            ensureHealthy();
            try {
                return guarded(context, operation.apply(activeTransport()));
            } catch (RuntimeException error) {
                return CompletableFuture.failedFuture(error);
            }
        }

        private SessionRecovery recovery() {
            return new SessionRecovery(
                    configuration,
                    currentUri,
                    currentOrigin,
                    documentRevision.get()
            );
        }

        private void closePublisher() {
            events.close();
        }
    }

    private JcefHostProtocol.Hello negotiate(Transport candidate) {
        JcefHostProtocol.Hello hello = await(
                candidate.hello(),
                handshakeTimeout,
                "JCEF host protocol negotiation"
        );
        if (hello.protocolVersion() != JcefHostProtocol.VERSION) {
            throw new IllegalStateException(
                    "Unsupported JCEF host protocol " + hello.protocolVersion()
                            + "; expected " + JcefHostProtocol.VERSION
            );
        }
        return hello;
    }

    private void verifyCompatible(JcefHostProtocol.Hello hello) {
        if (!hostEngineId.equals(hello.engineId())) {
            throw new IllegalStateException(
                    "Restarted JCEF host identity changed from " + hostEngineId
                            + " to " + hello.engineId()
            );
        }
        if (!capabilities.equals(capabilities(hello))) {
            throw new IllegalStateException("Restarted JCEF host capability declaration changed");
        }
    }

    private static Capabilities capabilities(JcefHostProtocol.Hello hello) {
        return new Capabilities(
                true,
                false,
                hello.nativeSharedSurfaces(),
                hello.semanticSnapshots(),
                hello.siteIsolation(),
                hello.schemes()
        );
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

    private static void closeQuietly(Transport candidate) {
        if (candidate == null) return;
        try {
            candidate.close();
        } catch (RuntimeException ignored) {
            // Preserve the primary lifecycle result.
        }
    }
}
