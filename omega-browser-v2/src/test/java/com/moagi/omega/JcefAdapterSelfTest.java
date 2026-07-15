package com.moagi.omega;

import com.moagi.omega.adapter.jcef.JcefBrowserEngine;
import com.moagi.omega.adapter.jcef.JcefHostProtocol;
import com.moagi.omega.adapter.jcef.JcefHostProtocol.HostEvent;
import com.moagi.omega.api.*;
import com.moagi.omega.api.Engine.CapabilityResolution;
import com.moagi.omega.api.Engine.Event;
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
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Flow;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;

/** Contract test for the process-hosted JCEF adapter without bundled Chromium binaries. */
public final class JcefAdapterSelfTest {
    private JcefAdapterSelfTest() {}

    public static void run() throws Exception {
        var transport = new FakeJcefTransport();
        try (var engine = JcefBrowserEngine.connect(
                transport,
                Duration.ofSeconds(2),
                Duration.ofSeconds(3)
        )) {
            require(engine.id().equals("jcef-contract-host"), "JCEF host identity was not negotiated");
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
            var profile = Files.createTempDirectory("omega-jcef-adapter-test-");
            BlockingQueue<BrowserEvent> observed = new LinkedBlockingQueue<>();

            try (var kernel = new BrowserKernel(selector, broker, profile)) {
                subscribe(kernel, observed);
                UUID session = kernel.openSession(
                        false,
                        800,
                        500,
                        URI.create("https://adapter.example/index.html")
                );
                kernel.awaitIdle(session).toCompletableFuture().get(3, TimeUnit.SECONDS);

                Snapshot snapshot = kernel.snapshot(session)
                        .toCompletableFuture().get(2, TimeUnit.SECONDS);
                require(snapshot.documentUri().getHost().equals("adapter.example"),
                        "JCEF snapshot did not preserve the committed URI");
                require(!snapshot.nodes().isEmpty(), "JCEF semantic snapshot was empty");

                awaitEvent(observed, event -> isEngineEvent(
                        event, session, Event.NavigationCommitted.class), Duration.ofSeconds(2));
                awaitEvent(observed, event -> isEngineEvent(
                        event, session, Event.FrameReady.class), Duration.ofSeconds(2));
                awaitEvent(observed, event -> isEngineEvent(
                        event, session, Event.SnapshotReady.class), Duration.ofSeconds(2));

                UUID capabilityRequest = transport.requestCapability(
                        session,
                        Capability.GEOLOCATION,
                        "HTML geolocation request"
                );
                CapabilityResolution resolution = transport.resolution(capabilityRequest)
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
                        "JCEF single-use authority was not consumed before delivery");

                var reload = kernel.dispatch(session, new BrowserCommand.Reload())
                        .toCompletableFuture().get(3, TimeUnit.SECONDS);
                require(reload.state() == TransactionJournal.State.COMMITTED,
                        "JCEF reload did not commit");

                var slowNavigation = kernel.dispatch(
                        session,
                        new BrowserCommand.Navigate(URI.create("https://adapter.example/slow"))
                );
                awaitEvent(observed, event -> event instanceof BrowserEvent.EngineEvent engineEvent
                                && engineEvent.sessionId().equals(session)
                                && engineEvent.event() instanceof Event.NavigationStarted started
                                && started.uri().getPath().equals("/slow"),
                        Duration.ofSeconds(2));

                var stop = kernel.dispatch(session, new BrowserCommand.Stop())
                        .toCompletableFuture().get(2, TimeUnit.SECONDS);
                var cancelledNavigation = slowNavigation.toCompletableFuture()
                        .get(2, TimeUnit.SECONDS);
                require(stop.state() == TransactionJournal.State.COMMITTED,
                        "out-of-band JCEF cancellation did not commit");
                require(stop.committedRevision() == stop.parentRevision(),
                        "control-lane stop incorrectly advanced document revision");
                require(cancelledNavigation.state() == TransactionJournal.State.FAILED,
                        "cancelled JCEF navigation was not recorded as failed");

                transport.crash(session, "simulated renderer termination", true);
                awaitEvent(observed, event -> event instanceof BrowserEvent.EngineEvent engineEvent
                                && engineEvent.sessionId().equals(session)
                                && engineEvent.event() instanceof Event.Crashed crash
                                && crash.recoverable(),
                        Duration.ofSeconds(2));
                require(engine.healthy(),
                        "recoverable renderer crash incorrectly killed the JCEF host adapter");

                kernel.closeSession(session);
                require(transport.closedSession(session),
                        "JCEF session shutdown was not forwarded to the host transport");
            }
        }
    }

    private static void subscribe(
            BrowserKernel kernel,
            BlockingQueue<BrowserEvent> observed
    ) {
        kernel.events().subscribe(new Flow.Subscriber<>() {
            @Override
            public void onSubscribe(Flow.Subscription subscription) {
                subscription.request(Long.MAX_VALUE);
            }

            @Override
            public void onNext(BrowserEvent item) {
                observed.offer(item);
            }

            @Override
            public void onError(Throwable throwable) {
                observed.offer(new BrowserEvent.SessionChanged(
                        new UUID(0, 0), "EVENT_ERROR " + throwable, "test"
                ));
            }

            @Override
            public void onComplete() {}
        });
    }

    private static boolean isEngineEvent(
            BrowserEvent event,
            UUID sessionId,
            Class<? extends Event> eventType
    ) {
        return event instanceof BrowserEvent.EngineEvent engineEvent
                && engineEvent.sessionId().equals(sessionId)
                && eventType.isInstance(engineEvent.event());
    }

    private static BrowserEvent awaitEvent(
            BlockingQueue<BrowserEvent> observed,
            Predicate<BrowserEvent> predicate,
            Duration timeout
    ) throws Exception {
        long deadline = System.nanoTime() + timeout.toNanos();
        while (System.nanoTime() < deadline) {
            long remaining = deadline - System.nanoTime();
            BrowserEvent event = observed.poll(
                    Math.max(1, TimeUnit.NANOSECONDS.toMillis(remaining)),
                    TimeUnit.MILLISECONDS
            );
            if (event != null && predicate.test(event)) return event;
        }
        throw new AssertionError("Timed out waiting for JCEF adapter event");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    private static final class FakeJcefTransport implements JcefHostProtocol.Transport {
        private final SubmissionPublisher<HostEvent> events = new SubmissionPublisher<>();
        private final Map<UUID, SessionState> sessions = new ConcurrentHashMap<>();
        private final Map<UUID, CompletableFuture<CapabilityResolution>> resolutions =
                new ConcurrentHashMap<>();
        private final Set<UUID> closedSessions = ConcurrentHashMap.newKeySet();
        private final AtomicBoolean healthy = new AtomicBoolean(true);

        @Override
        public CompletionStage<JcefHostProtocol.Hello> hello() {
            return CompletableFuture.completedFuture(new JcefHostProtocol.Hello(
                    JcefHostProtocol.VERSION,
                    "contract-host",
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
        public CompletionStage<Void> navigate(UUID sessionId, URI uri) {
            SessionState state = requireSession(sessionId);
            state.pendingUri = uri;
            events.submit(new HostEvent.NavigationStarted(sessionId, uri));

            if ("/slow".equals(uri.getPath())) {
                CompletableFuture<Void> pending = new CompletableFuture<>();
                state.pendingNavigation = pending;
                return pending;
            }

            commit(state, uri);
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<Void> reload(UUID sessionId) {
            return navigate(sessionId, requireSession(sessionId).currentUri);
        }

        @Override
        public CompletionStage<Void> stop(UUID sessionId) {
            SessionState state = requireSession(sessionId);
            CompletableFuture<Void> pending = state.pendingNavigation;
            if (pending != null && !pending.isDone()) {
                URI cancelledUri = state.pendingUri;
                events.submit(new HostEvent.NavigationFailed(
                        sessionId, cancelledUri, "cancelled by kernel control lane"
                ));
                pending.completeExceptionally(new CancellationException("navigation cancelled"));
                state.pendingNavigation = null;
            }
            events.submit(new HostEvent.Status(sessionId, "stop acknowledged"));
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<Void> execute(
                UUID sessionId,
                long nodeId,
                Action action,
                Map<String, String> arguments
        ) {
            requireSession(sessionId);
            events.submit(new HostEvent.Status(
                    sessionId, "semantic action " + action + " on node " + nodeId
            ));
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<Snapshot> snapshot(UUID sessionId) {
            return CompletableFuture.completedFuture(requireSession(sessionId).snapshot);
        }

        @Override
        public CompletionStage<Void> resolveCapability(
                UUID sessionId,
                CapabilityResolution resolution
        ) {
            requireSession(sessionId);
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
            SessionState state = requireSession(sessionId);
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
            CompletableFuture<CapabilityResolution> resolution = resolutions.get(requestId);
            if (resolution == null) {
                return CompletableFuture.failedFuture(
                        new IllegalArgumentException("Unknown capability request " + requestId)
                );
            }
            return resolution;
        }

        void crash(UUID sessionId, String reason, boolean recoverable) {
            requireSession(sessionId);
            events.submit(new HostEvent.Crashed(sessionId, reason, recoverable));
        }

        boolean closedSession(UUID sessionId) {
            return closedSessions.contains(sessionId);
        }

        private void commit(SessionState state, URI uri) {
            state.currentUri = uri;
            long revision = state.revision.incrementAndGet();
            long frameSequence = state.frameSequence.incrementAndGet();
            state.snapshot = snapshot(uri, revision, frameSequence);
            events.submit(new HostEvent.SnapshotReady(
                    state.configuration.sessionId(), state.snapshot
            ));
            events.submit(new HostEvent.FrameReady(
                    state.configuration.sessionId(),
                    new CpuFrame(
                            frameSequence,
                            2,
                            2,
                            new int[]{0xFF102030, 0xFF203040, 0xFF304050, 0xFF405060},
                            Geometry.fullDamage(2, 2)
                    )
            ));
            events.submit(new HostEvent.NavigationCommitted(
                    state.configuration.sessionId(),
                    uri,
                    Origin.from(uri),
                    "JCEF contract page"
            ));
            state.pendingNavigation = null;
        }

        private SessionState requireSession(UUID sessionId) {
            SessionState state = sessions.get(sessionId);
            if (state == null) throw new IllegalArgumentException("Unknown fake JCEF session " + sessionId);
            return state;
        }

        private static Snapshot snapshot(URI uri, long revision, long frameSequence) {
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
                    frameSequence,
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

            SessionState(SessionConfiguration configuration) {
                this.configuration = configuration;
            }
        }
    }
}
