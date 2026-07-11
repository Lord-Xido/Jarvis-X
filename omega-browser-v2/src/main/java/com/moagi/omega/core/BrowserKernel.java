package com.moagi.omega.core;

import com.moagi.omega.api.*;
import com.moagi.omega.api.BrowserEvent.*;
import com.moagi.omega.api.Engine.*;
import com.moagi.omega.core.TransactionJournal.State;
import com.moagi.omega.core.TransactionJournal.TransactionSnapshot;

import java.net.URI;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Trusted browser kernel. It owns session identity, command authorization,
 * capability grants, transaction state and engine event forwarding.
 */
public final class BrowserKernel implements AutoCloseable {
    private final EngineSelector selector;
    private final CapabilityBroker capabilityBroker;
    private final TransactionJournal journal;
    private final ExecutorService tasks = Executors.newVirtualThreadPerTaskExecutor();
    private final SubmissionPublisher<BrowserEvent> events = new SubmissionPublisher<>(tasks, 512);
    private final Map<UUID, KernelSession> sessions = new ConcurrentHashMap<>();
    private final AtomicLong revision = new AtomicLong();

    public BrowserKernel(
            EngineSelector selector,
            CapabilityBroker capabilityBroker,
            Path profileDirectory
    ) {
        this.selector = Objects.requireNonNull(selector);
        this.capabilityBroker = Objects.requireNonNull(capabilityBroker);
        this.journal = new TransactionJournal(profileDirectory.resolve("journal.tsv"));
    }

    public Flow.Publisher<BrowserEvent> events() { return events; }
    public TransactionJournal journal() { return journal; }

    public UUID openSession(boolean privateMode, int viewportWidth, int viewportHeight, URI initialUri) {
        EngineSelector.Selection selection = selector.select(initialUri, privateMode);
        UUID id = UUID.randomUUID();
        SessionConfiguration configuration = new SessionConfiguration(
                id, privateMode, viewportWidth, viewportHeight,
                privateMode ? "private-" + id : "default"
        );
        EngineSession engineSession = selection.engine().createSession(configuration);
        KernelSession kernelSession = new KernelSession(id, privateMode, selection.engine(), engineSession);
        sessions.put(id, kernelSession);
        subscribe(kernelSession);
        events.submit(new SessionChanged(id, "OPEN", selection.engine().id()));
        dispatch(id, new BrowserCommand.Navigate(initialUri));
        return id;
    }

    public CompletionStage<TransactionSnapshot> dispatch(UUID sessionId, BrowserCommand command) {
        KernelSession session = requireSession(sessionId);
        Origin origin = session.engineSession.currentOrigin();
        long parent = revision.get();
        TransactionSnapshot created = journal.create(sessionId, parent, command, origin);
        events.submit(new TransactionChanged(sessionId, created));

        return CompletableFuture.supplyAsync(() -> executeTransaction(session, created, command), tasks);
    }

    public CompletionStage<com.moagi.omega.api.SemanticScene.Snapshot> snapshot(UUID sessionId) {
        return requireSession(sessionId).engineSession.snapshot();
    }

    public void closeSession(UUID sessionId) {
        KernelSession session = sessions.remove(sessionId);
        if (session == null) return;
        capabilityBroker.revokeSession(sessionId);
        session.engineSession.close();
        events.submit(new SessionChanged(sessionId, "CLOSED", session.engine.id()));
    }

    private TransactionSnapshot executeTransaction(
            KernelSession session,
            TransactionSnapshot transaction,
            BrowserCommand command
    ) {
        TransactionSnapshot current = transition(transaction, State.VALIDATING, -1, "validating command");
        try {
            validate(session, command);
            current = transition(current, State.AUTHORIZED, -1, "policy authorized");
            current = transition(current, State.EXECUTING, -1, "engine execution started");

            CompletionStage<Void> operation = execute(session, command);
            operation.toCompletableFuture().join();

            long committedRevision = revision.incrementAndGet();
            return transition(current, State.COMMITTED, committedRevision, "committed");
        } catch (CompletionException ex) {
            Throwable cause = ex.getCause() == null ? ex : ex.getCause();
            return transition(current, State.FAILED, -1, cause.getMessage());
        } catch (SecurityException | IllegalArgumentException ex) {
            return transition(current, State.REJECTED, -1, ex.getMessage());
        } catch (RuntimeException ex) {
            return transition(current, State.FAILED, -1, ex.toString());
        }
    }

    private void validate(KernelSession session, BrowserCommand command) {
        if (command instanceof BrowserCommand.Navigate navigation) {
            String scheme = navigation.uri().getScheme();
            if (scheme == null || !session.engine.capabilities().schemes().contains(scheme.toLowerCase())) {
                throw new SecurityException("Engine does not permit URI scheme: " + scheme);
            }
        }
        if (command instanceof BrowserCommand.SemanticAction action && action.nodeId() < 0) {
            throw new IllegalArgumentException("Semantic node id must be non-negative");
        }
    }

    private CompletionStage<Void> execute(KernelSession session, BrowserCommand command) {
        if (command instanceof BrowserCommand.Navigate navigation) {
            return session.engineSession.navigate(navigation.uri());
        }
        if (command instanceof BrowserCommand.Reload) return session.engineSession.reload();
        if (command instanceof BrowserCommand.Stop) return session.engineSession.stop();
        if (command instanceof BrowserCommand.SemanticAction action) {
            return session.engineSession.execute(action.nodeId(), action.action(), action.arguments());
        }
        if (command instanceof BrowserCommand.RequestCapability request) {
            Origin origin = session.engineSession.currentOrigin();
            var token = capabilityBroker.request(
                    session.id, origin, request.capability(), request.singleUse(),
                    "explicit browser command"
            );
            if (token.isEmpty()) {
                events.submit(new CapabilityChanged(session.id, null, "DENIED " + request.capability()));
                throw new SecurityException("Capability denied: " + request.capability());
            }
            events.submit(new CapabilityChanged(session.id, token.get(), "GRANTED " + request.capability()));
            return CompletableFuture.completedFuture(null);
        }
        if (command instanceof BrowserCommand.SetSpatialMode mode) {
            session.spatialMode = mode.enabled();
            return CompletableFuture.completedFuture(null);
        }
        return CompletableFuture.failedFuture(new IllegalArgumentException("Unsupported command"));
    }

    private TransactionSnapshot transition(
            TransactionSnapshot previous,
            State state,
            long committedRevision,
            String detail
    ) {
        TransactionSnapshot next = journal.transition(previous, state, committedRevision, detail);
        events.submit(new TransactionChanged(previous.sessionId(), next));
        return next;
    }

    private void subscribe(KernelSession session) {
        session.engineSession.events().subscribe(new Flow.Subscriber<>() {
            private Flow.Subscription subscription;

            @Override
            public void onSubscribe(Flow.Subscription subscription) {
                this.subscription = subscription;
                subscription.request(Long.MAX_VALUE);
            }

            @Override
            public void onNext(Event item) {
                events.submit(new EngineEvent(session.id, item));
                if (item instanceof Event.CapabilityRequested request) {
                    var token = capabilityBroker.request(
                            session.id, request.origin(), request.capability(), true, request.rationale()
                    );
                    events.submit(new CapabilityChanged(
                            session.id,
                            token.orElse(null),
                            token.isPresent() ? "GRANTED " + request.capability() : "DENIED " + request.capability()
                    ));
                }
            }

            @Override public void onError(Throwable throwable) {
                events.submit(new EngineEvent(session.id,
                        new Event.Crashed(throwable.toString(), true)));
            }

            @Override public void onComplete() {}
        });
    }

    private KernelSession requireSession(UUID id) {
        KernelSession session = sessions.get(id);
        if (session == null) throw new IllegalArgumentException("Unknown browser session: " + id);
        return session;
    }

    @Override
    public void close() {
        for (UUID sessionId : sessions.keySet()) closeSession(sessionId);
        selector.engines().forEach(BrowserEngine::close);
        events.close();
        tasks.close();
    }

    private static final class KernelSession {
        final UUID id;
        final boolean privateMode;
        final BrowserEngine engine;
        final EngineSession engineSession;
        volatile boolean spatialMode;

        KernelSession(UUID id, boolean privateMode, BrowserEngine engine, EngineSession engineSession) {
            this.id = id;
            this.privateMode = privateMode;
            this.engine = engine;
            this.engineSession = engineSession;
        }
    }
}
