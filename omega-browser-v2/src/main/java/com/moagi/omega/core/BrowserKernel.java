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
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

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
        Objects.requireNonNull(initialUri, "initialUri");
        if (viewportWidth <= 0 || viewportHeight <= 0) {
            throw new IllegalArgumentException("Viewport dimensions must be positive");
        }

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

    /**
     * Enqueues state-mutating commands onto the session's single transactional
     * chain. Stop is a journaled control-lane command so it can interrupt an
     * in-flight engine operation rather than waiting behind it.
     */
    public CompletionStage<TransactionSnapshot> dispatch(UUID sessionId, BrowserCommand command) {
        Objects.requireNonNull(command, "command");
        KernelSession session = requireSession(sessionId);
        if (command instanceof BrowserCommand.Stop) {
            return dispatchControl(session, command);
        }
        return session.enqueue(() -> {
            Origin origin = session.engineSession.currentOrigin();
            long parent = session.revision.get();
            TransactionSnapshot created = journal.create(sessionId, parent, command, origin);
            events.submit(new TransactionChanged(sessionId, created));
            return executeTransaction(session, created, command);
        }, tasks);
    }

    private CompletionStage<TransactionSnapshot> dispatchControl(
            KernelSession session,
            BrowserCommand command
    ) {
        return CompletableFuture.supplyAsync(() -> {
            Origin origin = session.engineSession.currentOrigin();
            long parent = session.revision.get();
            TransactionSnapshot created = journal.create(session.id, parent, command, origin);
            events.submit(new TransactionChanged(session.id, created));
            TransactionSnapshot current = transition(
                    created, State.VALIDATING, -1, "validating control command"
            );
            try {
                validate(session, command);
                current = transition(current, State.AUTHORIZED, -1, "control policy authorized");
                current = transition(current, State.EXECUTING, -1, "control execution started");
                execute(session, command).toCompletableFuture().join();
                return transition(
                        current,
                        State.COMMITTED,
                        session.revision.get(),
                        "control command committed without revision advance"
                );
            } catch (CompletionException ex) {
                Throwable cause = ex.getCause() == null ? ex : ex.getCause();
                return transition(current, State.FAILED, -1, cause.getMessage());
            } catch (SecurityException | IllegalArgumentException ex) {
                return transition(current, State.REJECTED, -1, ex.getMessage());
            } catch (RuntimeException ex) {
                return transition(current, State.FAILED, -1, ex.toString());
            }
        }, tasks);
    }

    /** Returns a stage that completes when all currently queued state work is done. */
    public CompletionStage<Void> awaitIdle(UUID sessionId) {
        return requireSession(sessionId).idle();
    }

    public CompletionStage<com.moagi.omega.api.SemanticScene.Snapshot> snapshot(UUID sessionId) {
        return requireSession(sessionId).engineSession.snapshot();
    }

    public void closeSession(UUID sessionId) {
        KernelSession session = sessions.remove(sessionId);
        if (session == null) return;
        session.closed = true;
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

            long observedRevision = session.revision.get();
            if (transaction.parentRevision() != observedRevision) {
                throw new IllegalArgumentException(
                        "Stale parent revision " + transaction.parentRevision()
                                + "; current session revision is " + observedRevision
                );
            }

            current = transition(current, State.AUTHORIZED, -1, "policy authorized");
            current = transition(current, State.EXECUTING, -1, "engine execution started");

            CompletionStage<Void> operation = execute(session, command);
            operation.toCompletableFuture().join();

            long committedRevision = observedRevision + 1;
            if (!session.revision.compareAndSet(observedRevision, committedRevision)) {
                throw new IllegalArgumentException("Session revision changed during transaction execution");
            }
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
        if (session.closed) throw new SecurityException("Browser session is closed");

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
                    resolveCapabilityRequest(session, request);
                }
            }

            @Override public void onError(Throwable throwable) {
                events.submit(new EngineEvent(session.id,
                        new Event.Crashed(throwable.toString(), true)));
            }

            @Override public void onComplete() {}
        });
    }

    private void resolveCapabilityRequest(KernelSession session, Event.CapabilityRequested request) {
        Origin currentOrigin = session.engineSession.currentOrigin();
        Optional<CapabilityToken> token = Optional.empty();
        String detail;

        if (session.closed) {
            detail = "session closed";
        } else if (!currentOrigin.equals(request.origin())) {
            detail = "request origin does not match the active session origin";
        } else {
            token = capabilityBroker.request(
                    session.id,
                    request.origin(),
                    request.capability(),
                    true,
                    request.rationale()
            );
            detail = token.isPresent() ? "broker approved" : "broker denied";
        }

        boolean granted = token.isPresent() && capabilityBroker.validateAndConsume(
                token.get().tokenId(),
                session.id,
                request.origin(),
                request.capability()
        );
        CapabilityToken grantedToken = granted ? token.get() : null;
        CapabilityResolution resolution = new CapabilityResolution(
                request.requestId(),
                request.capability(),
                request.origin(),
                granted,
                granted ? grantedToken.tokenId() : null,
                granted ? detail : detail + "; no authority released"
        );

        session.engineSession.resolveCapability(resolution).whenComplete((ignored, error) -> {
            if (error != null) {
                events.submit(new EngineEvent(session.id,
                        new Event.Status("Capability resolution delivery failed: " + error)));
            }
        });
        events.submit(new CapabilityChanged(
                session.id,
                grantedToken,
                granted ? "GRANTED " + request.capability() : "DENIED " + request.capability()
        ));
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
        final AtomicLong revision = new AtomicLong();
        private CompletableFuture<Void> tail = CompletableFuture.completedFuture(null);
        volatile boolean spatialMode;
        volatile boolean closed;

        KernelSession(UUID id, boolean privateMode, BrowserEngine engine, EngineSession engineSession) {
            this.id = id;
            this.privateMode = privateMode;
            this.engine = engine;
            this.engineSession = engineSession;
        }

        synchronized CompletionStage<TransactionSnapshot> enqueue(
                Supplier<TransactionSnapshot> operation,
                Executor executor
        ) {
            if (closed) {
                return CompletableFuture.failedFuture(
                        new IllegalArgumentException("Browser session is closed: " + id)
                );
            }
            CompletableFuture<TransactionSnapshot> result = tail
                    .handle((ignored, error) -> null)
                    .thenApplyAsync(ignored -> operation.get(), executor);
            tail = result.handle((ignored, error) -> null);
            return result;
        }

        synchronized CompletionStage<Void> idle() {
            return tail.thenApply(ignored -> null);
        }
    }
}
