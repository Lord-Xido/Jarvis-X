package com.moagi.omega;

import com.moagi.omega.adapter.jcef.JcefBrowserEngine;
import com.moagi.omega.adapter.jcef.JcefHostProtocol;
import com.moagi.omega.adapter.jcef.JcefHostProtocol.HostEvent;
import com.moagi.omega.adapter.jcef.JcefHostProtocol.SessionRecovery;
import com.moagi.omega.api.*;
import com.moagi.omega.api.Engine.CapabilityResolution;
import com.moagi.omega.api.Engine.Event;
import com.moagi.omega.api.Engine.OperationContext;
import com.moagi.omega.api.Engine.SessionConfiguration;
import com.moagi.omega.api.Geometry.Rect;
import com.moagi.omega.api.Geometry.Transform3D;
import com.moagi.omega.api.SemanticScene.Action;
import com.moagi.omega.api.SemanticScene.Node;
import com.moagi.omega.api.SemanticScene.Snapshot;
import com.moagi.omega.api.Surface.CpuFrame;
import com.moagi.omega.core.BrowserKernel;
import com.moagi.omega.core.CapabilityBroker;
import com.moagi.omega.core.EngineSelector;
import com.moagi.omega.core.TransactionJournal;

import java.net.URI;
import java.nio.file.Files;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Flow;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;

/** Contract test for the process-hosted JCEF adapter without bundled Chromium binaries. */
public final class JcefAdapterSelfTest {
    private JcefAdapterSelfTest() {}

    public static void run() throws Exception {
        var firstTransport = new FakeJcefTransport("contract-host");
        try (var engine = JcefBrowserEngine.connect(
                firstTransport,
                Duration.ofSeconds(2),
                Duration.ofSeconds(3)
        )) {
            require(engine.id().equals("jcef-contract-host"),
                    "JCEF host identity was not negotiated");
            require(!engine.capabilities().nativeSharedSurfaces(),
                    "adapter claimed unverified native shared surfaces");
            require(!engine.capabilities().siteIsolation(),
                    "adapter claimed unverified site isolation");

            var selector = new EngineSelector();
            selector.register(engine);
            var broker = new CapabilityBroker((session, origin, capability, rationale) ->
                    capability == Capability.GEOLOCATION
                            ? CapabilityBroker.Decision.ALLOW
                            : CapabilityBroker.Decision.DENY);
            var recorder = new EventRecorder();

            try (var kernel = new BrowserKernel(
                    selector,
                    broker,
                    Files.createTempDirectory("omega-jcef-adapter-test-")
            )) {
                kernel.events().subscribe(recorder);
                UUID session = kernel.openSession(
                        false,
                        800,
                        500,
                        URI.create("https://adapter.example/index.html")
                );
                kernel.awaitIdle(session).toCompletableFuture().get(3, TimeUnit.SECONDS);

                Snapshot snapshot = kernel.snapshot(session)
                        .toCompletableFuture().get(2, TimeUnit.SECONDS);
                require("adapter.example".equals(snapshot.documentUri().getHost()),
                        "JCEF snapshot did not preserve the committed URI");
                require(!snapshot.nodes().isEmpty(), "JCEF semantic snapshot was empty");

                recorder.awaitCorrelated(session, Event.SnapshotReady.class, Duration.ofSeconds(2));
                recorder.awaitCorrelated(session, Event.FrameReady.class, Duration.ofSeconds(2));
                recorder.awaitCorrelated(session, Event.NavigationCommitted.class, Duration.ofSeconds(2));

                UUID requestId = firstTransport.requestCapability(
                        session, Capability.GEOLOCATION, "HTML geolocation request"
                );
                CapabilityResolution resolution = firstTransport.resolution(requestId)
                        .toCompletableFuture().get(2, TimeUnit.SECONDS);
                require(resolution.granted(), "JCEF capability request was not granted");
                require(resolution.origin().equals(Origin.from(
                                URI.create("https://adapter.example/index.html"))),
                        "JCEF capability decision was not origin-bound");
                require(!broker.validateAndConsume(
                                resolution.tokenId(),
                                session,
                                resolution.origin(),
                                resolution.capability()),
                        "single-use authority was not consumed before host delivery");

                var reload = kernel.dispatch(session, new BrowserCommand.Reload())
                        .toCompletableFuture().get(3, TimeUnit.SECONDS);
                require(reload.state() == TransactionJournal.State.COMMITTED,
                        "JCEF reload did not commit");
                Event.Correlated reloadCommit = recorder.awaitCorrelated(
                        session,
                        event -> event.event() instanceof Event.NavigationCommitted
                                && event.transactionId().equals(reload.transactionId()),
                        Duration.ofSeconds(2)
                );
                require(reloadCommit.documentRevision() >= 2,
                        "reload callback did not carry a document revision");

                var slowNavigation = kernel.dispatch(
                        session,
                        new BrowserCommand.Navigate(URI.create("https://adapter.example/slow"))
                );
                Event.Correlated slowStarted = recorder.awaitCorrelated(
                        session,
                        event -> event.event() instanceof Event.NavigationStarted started
                                && "/slow".equals(started.uri().getPath()),
                        Duration.ofSeconds(2)
                );

                var stop = kernel.dispatch(session, new BrowserCommand.Stop())
                        .toCompletableFuture().get(2, TimeUnit.SECONDS);
                var cancelled = slowNavigation.toCompletableFuture().get(2, TimeUnit.SECONDS);
                require(stop.state() == TransactionJournal.State.COMMITTED,
                        "out-of-band JCEF stop did not commit");
                require(stop.parentRevision() == stop.committedRevision(),
                        "stop advanced the document revision");
                require(cancelled.state() == TransactionJournal.State.FAILED,
                        "cancelled navigation was not journaled as failed");
                require(slowStarted.transactionId().equals(cancelled.transactionId()),
                        "navigation callback lost kernel transaction correlation");
                recorder.awaitCorrelated(
                        session,
                        event -> event.event() instanceof Event.NavigationFailed
                                && event.transactionId().equals(cancelled.transactionId()),
                        Duration.ofSeconds(2)
                );

                firstTransport.crash(session, "simulated renderer termination", true);
                recorder.await(event -> event instanceof BrowserEvent.EngineEvent engineEvent
                                && engineEvent.sessionId().equals(session)
                                && engineEvent.event() instanceof Event.Crashed crash
                                && crash.recoverable(),
                        Duration.ofSeconds(2));
                require(engine.healthy(),
                        "recoverable renderer crash killed the JCEF host adapter");

                var replacementTransport = new FakeJcefTransport("contract-host");
                engine.recover(replacementTransport);
                require(engine.healthy(), "adapter was not healthy after host recovery");
                require(replacementTransport.recoveredSession(session),
                        "live session was not rehydrated on replacement host");
                require(replacementTransport.currentUri(session).equals(
                                URI.create("https://adapter.example/index.html")),
                        "session URI was not preserved across host recovery");

                var recoveredNavigation = kernel.dispatch(
                        session,
                        new BrowserCommand.Navigate(URI.create("https://adapter.example/recovered"))
                ).toCompletableFuture().get(3, TimeUnit.SECONDS);
                require(recoveredNavigation.state() == TransactionJournal.State.COMMITTED,
                        "navigation through recovered host did not commit");
                recorder.awaitCorrelated(
                        session,
                        event -> event.event() instanceof Event.NavigationCommitted committed
                                && "/recovered".equals(committed.uri().getPath())
                                && event.transactionId().equals(recoveredNavigation.transactionId()),
                        Duration.ofSeconds(2)
                );

                kernel.closeSession(session);
                require(replacementTransport.closedSession(session),
                        "JCEF session shutdown was not forwarded to replacement host");
            }
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static final class EventRecorder implements Flow.Subscriber<BrowserEvent> {
        private final List<BrowserEvent> events = new CopyOnWriteArrayList<>();

        @Override
        public void onSubscribe(Flow.Subscription subscription) {
            subscription.request(Long.MAX_VALUE);
        }

        @Override
        public void onNext(BrowserEvent item) {
            events.add(item);
        }

        @Override
        public void onError(Throwable throwable) {
            throw new AssertionError("Kernel event stream failed", throwable);
        }

        @Override
        public void onComplete() {}

        BrowserEvent await(Predicate<BrowserEvent> predicate, Duration timeout)
                throws InterruptedException {
            long deadline = System.nanoTime() + timeout.toNanos();
            while (System.nanoTime() < deadline) {
                for (BrowserEvent event : events) {
                    if (predicate.test(event)) return event;
                }
                Thread.sleep(10);
            }
            throw new AssertionError("Timed out waiting for JCEF adapter event");
        }

        Event.Correlated awaitCorrelated(
                UUID sessionId,
                Class<? extends Event> eventType,
                Duration timeout
        ) throws InterruptedException {
            return awaitCorrelated(
                    sessionId,
                    correlated -> eventType.isInstance(correlated.event()),
                    timeout
            );
        }

        Event.Correlated awaitCorrelated(
                UUID sessionId,
                Predicate<Event.Correlated> predicate,
                Duration timeout
        ) throws InterruptedException {
            BrowserEvent found = await(event ->
                    event instanceof BrowserEvent.EngineEvent engineEvent
                            && engineEvent.sessionId().equals(sessionId)
                            && engineEvent.event() instanceof Event.Correlated correlated
                            && predicate.test(correlated),
                    timeout
            );
            return (Event.Correlated) ((BrowserEvent.EngineEvent) found).event();
        }
    }

    private static final class FakeJcefTransport implements JcefHostProtocol.Transport {
        private final String engineId;
        private final SubmissionPublisher<HostEvent> events = new SubmissionPublisher<>();
        private final Map<UUID, SessionState> sessions = new ConcurrentHashMap<>();
        private final Map<UUID, CompletableFuture<CapabilityResolution>> resolutions =
                new ConcurrentHashMap<>();
        private final Set<UUID> closedSessions = ConcurrentHashMap.newKeySet();
        private final Set<UUID> recoveredSessions = ConcurrentHashMap.newKeySet();
        private final AtomicBoolean healthy = new AtomicBoolean(true);

        private FakeJcefTransport(String engineId) {
            this.engineId = engineId;
        }

        @Override
        public CompletionStage<JcefHostProtocol.Hello> hello() {
            return CompletableFuture.completedFuture(new JcefHostProtocol.Hello(
                    JcefHostProtocol.VERSION,
                    engineId,
                    "test-1.0",
                    false,
                    true,
                    false,
                    Set.of("https", "http", "about")
            ));
        }

        @Override
        public CompletionStage<Void> createSession(SessionConfiguration configuration) {
            sessions.put(configuration.sessionId(), new SessionState(configuration));
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<Void> recoverSession(SessionRecovery recovery) {
            SessionState state = new SessionState(recovery.configuration());
            state.currentUri = recovery.currentUri();
            state.pendingUri = recovery.currentUri();
            state.revision.set(recovery.documentRevision());
            state.frameSequence.set(recovery.documentRevision());
            state.snapshot = snapshot(
                    recovery.currentUri(),
                    recovery.documentRevision(),
                    recovery.documentRevision()
            );
            sessions.put(recovery.configuration().sessionId(), state);
            recoveredSessions.add(recovery.configuration().sessionId());
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<Void> navigate(
                UUID sessionId,
                OperationContext context,
                URI uri
        ) {
            SessionState state = session(sessionId);
            state.pendingUri = uri;
            state.pendingContext = context;
            events.submit(new HostEvent.NavigationStarted(
                    sessionId, context.transactionId(), uri
            ));
            if ("/slow".equals(uri.getPath())) {
                state.pendingNavigation = new CompletableFuture<>();
                return state.pendingNavigation;
            }
            commit(state, context, uri);
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<Void> reload(UUID sessionId, OperationContext context) {
            return navigate(sessionId, context, session(sessionId).currentUri);
        }

        @Override
        public CompletionStage<Void> stop(UUID sessionId, OperationContext context) {
            SessionState state = session(sessionId);
            CompletableFuture<Void> pending = state.pendingNavigation;
            if (pending != null && !pending.isDone()) {
                OperationContext cancelledContext = state.pendingContext;
                events.submit(new HostEvent.NavigationFailed(
                        sessionId,
                        cancelledContext.transactionId(),
                        state.pendingUri,
                        "cancelled by kernel control lane"
                ));
                pending.completeExceptionally(new CancellationException("navigation cancelled"));
                state.pendingNavigation = null;
                state.pendingContext = null;
            }
            events.submit(new HostEvent.Status(
                    sessionId, context.transactionId(), "stop acknowledged"
            ));
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<Void> execute(
                UUID sessionId,
                OperationContext context,
                long nodeId,
                Action action,
                Map<String, String> arguments
        ) {
            session(sessionId);
            events.submit(new HostEvent.Status(
                    sessionId,
                    context.transactionId(),
                    "semantic action " + action + " on node " + nodeId
            ));
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<Snapshot> snapshot(UUID sessionId) {
            return CompletableFuture.completedFuture(session(sessionId).snapshot);
        }

        @Override
        public CompletionStage<Void> resolveCapability(
                UUID sessionId,
                CapabilityResolution resolution
        ) {
            session(sessionId);
            resolutions.computeIfAbsent(
                    resolution.requestId(), ignored -> new CompletableFuture<>()
            ).complete(resolution);
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<Void> closeSession(UUID sessionId) {
            sessions.remove(sessionId);
            closedSessions.add(sessionId);
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public Flow.Publisher<HostEvent> events() {
            return events;
        }

        @Override
        public boolean healthy() {
            return healthy.get();
        }

        @Override
        public void close() {
            healthy.set(false);
            events.close();
            sessions.clear();
        }

        UUID requestCapability(UUID sessionId, Capability capability, String rationale) {
            SessionState state = session(sessionId);
            UUID requestId = UUID.randomUUID();
            resolutions.put(requestId, new CompletableFuture<>());
            events.submit(new HostEvent.CapabilityRequested(
                    sessionId,
                    requestId,
                    capability,
                    Origin.from(state.currentUri),
                    rationale
            ));
            return requestId;
        }

        CompletionStage<CapabilityResolution> resolution(UUID requestId) {
            CompletableFuture<CapabilityResolution> future = resolutions.get(requestId);
            return future == null
                    ? CompletableFuture.failedFuture(
                            new IllegalArgumentException("Unknown request " + requestId))
                    : future;
        }

        void crash(UUID sessionId, String reason, boolean recoverable) {
            session(sessionId);
            events.submit(new HostEvent.Crashed(sessionId, reason, recoverable));
        }

        boolean closedSession(UUID sessionId) {
            return closedSessions.contains(sessionId);
        }

        boolean recoveredSession(UUID sessionId) {
            return recoveredSessions.contains(sessionId);
        }

        URI currentUri(UUID sessionId) {
            return session(sessionId).currentUri;
        }

        private SessionState session(UUID sessionId) {
            SessionState state = sessions.get(sessionId);
            if (state == null) {
                throw new IllegalArgumentException("Unknown fake JCEF session " + sessionId);
            }
            return state;
        }

        private void commit(SessionState state, OperationContext context, URI uri) {
            state.currentUri = uri;
            long revision = state.revision.incrementAndGet();
            long frame = state.frameSequence.incrementAndGet();
            state.snapshot = snapshot(uri, revision, frame);
            UUID sessionId = state.configuration.sessionId();
            UUID transactionId = context.transactionId();
            events.submit(new HostEvent.SnapshotReady(
                    sessionId, transactionId, state.snapshot
            ));
            events.submit(new HostEvent.FrameReady(
                    sessionId,
                    transactionId,
                    revision,
                    new CpuFrame(
                            frame,
                            2,
                            2,
                            new int[]{0xFF102030, 0xFF203040, 0xFF304050, 0xFF405060},
                            Geometry.fullDamage(2, 2)
                    )
            ));
            events.submit(new HostEvent.NavigationCommitted(
                    sessionId,
                    transactionId,
                    revision,
                    uri,
                    Origin.from(uri),
                    "JCEF contract page"
            ));
            state.pendingNavigation = null;
            state.pendingContext = null;
        }

        private static Snapshot snapshot(URI uri, long revision, long frame) {
            Node root = new Node(
                    0,
                    -1,
                    "document",
                    "JCEF contract document",
                    new Rect(0, 0, 800, 500),
                    Transform3D.identity(),
                    Map.of("uri", uri.toString()),
                    Set.of(Action.FOCUS),
                    List.of(),
                    0
            );
            return new Snapshot(
                    uri,
                    Origin.from(uri),
                    revision,
                    frame,
                    Instant.now(),
                    List.of(root)
            );
        }

        private static final class SessionState {
            final SessionConfiguration configuration;
            final AtomicLong revision = new AtomicLong();
            final AtomicLong frameSequence = new AtomicLong();
            volatile URI currentUri = URI.create("about:blank");
            volatile URI pendingUri = currentUri;
            volatile Snapshot snapshot = snapshot(currentUri, 0, 0);
            volatile CompletableFuture<Void> pendingNavigation;
            volatile OperationContext pendingContext;

            SessionState(SessionConfiguration configuration) {
                this.configuration = configuration;
            }
        }
    }
}
