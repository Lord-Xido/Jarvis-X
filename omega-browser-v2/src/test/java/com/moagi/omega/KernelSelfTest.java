package com.moagi.omega;

import com.moagi.omega.api.*;
import com.moagi.omega.core.*;
import com.moagi.omega.engine.mock.MockBrowserEngine;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

public final class KernelSelfTest {
    public static void main(String[] args) throws Exception {
        var profile = Files.createTempDirectory("omega-kernel-test-");
        var selector = new EngineSelector();
        var mockEngine = new MockBrowserEngine();
        selector.register(mockEngine);
        var broker = new CapabilityBroker((session, origin, capability, rationale) ->
                capability == Capability.GEOLOCATION
                        ? CapabilityBroker.Decision.ALLOW
                        : CapabilityBroker.Decision.DENY);

        try (var kernel = new BrowserKernel(selector, broker, profile)) {
            verifyViewportValidation(kernel);

            UUID session = kernel.openSession(false, 640, 420, URI.create("https://example.com/"));
            kernel.awaitIdle(session).toCompletableFuture().get(3, TimeUnit.SECONDS);

            var snapshot = kernel.snapshot(session).toCompletableFuture().get(2, TimeUnit.SECONDS);
            require(snapshot.nodes().size() >= 5, "semantic snapshot missing nodes");
            require(snapshot.origin().scheme().equals("https"), "origin normalization failed");

            var firstQueued = kernel.dispatch(session, new BrowserCommand.SetSpatialMode(true));
            var secondQueued = kernel.dispatch(session, new BrowserCommand.SetSpatialMode(false));
            var firstTx = firstQueued.toCompletableFuture().get(2, TimeUnit.SECONDS);
            var secondTx = secondQueued.toCompletableFuture().get(2, TimeUnit.SECONDS);
            require(firstTx.state() == TransactionJournal.State.COMMITTED,
                    "first queued transaction did not commit");
            require(secondTx.state() == TransactionJournal.State.COMMITTED,
                    "second queued transaction did not commit");
            require(secondTx.parentRevision() == firstTx.committedRevision(),
                    "session transaction chain was not serialized");
            require(secondTx.committedRevision() == firstTx.committedRevision() + 1,
                    "session revision did not advance exactly once");

            var semanticTx = kernel.dispatch(session,
                    new BrowserCommand.SemanticAction(4, SemanticScene.Action.CLICK, Map.of()))
                    .toCompletableFuture().get(3, TimeUnit.SECONDS);
            require(semanticTx.state() == TransactionJournal.State.COMMITTED,
                    "semantic action not committed");

            var capabilityTx = kernel.dispatch(session,
                    new BrowserCommand.RequestCapability(Capability.GEOLOCATION, true))
                    .toCompletableFuture().get(2, TimeUnit.SECONDS);
            require(capabilityTx.state() == TransactionJournal.State.COMMITTED,
                    "explicit capability transaction not committed");

            var engineCapabilityTx = kernel.dispatch(session,
                    new BrowserCommand.SemanticAction(5, SemanticScene.Action.FOCUS, Map.of()))
                    .toCompletableFuture().get(2, TimeUnit.SECONDS);
            require(engineCapabilityTx.state() == TransactionJournal.State.COMMITTED,
                    "engine capability request command did not commit");

            var resolution = mockEngine.capabilityResolution(session)
                    .toCompletableFuture().get(2, TimeUnit.SECONDS);
            require(resolution.granted(), "engine capability request was not granted");
            require(resolution.capability() == Capability.GEOLOCATION,
                    "engine received the wrong capability decision");
            require(!broker.validateAndConsume(
                            resolution.tokenId(), session, resolution.origin(), resolution.capability()),
                    "single-use engine capability token was not consumed by the kernel");

            var rejected = kernel.dispatch(session,
                    new BrowserCommand.Navigate(URI.create("file:///etc/passwd")))
                    .toCompletableFuture().get(2, TimeUnit.SECONDS);
            require(rejected.state() == TransactionJournal.State.REJECTED,
                    "forbidden scheme was not rejected");

            UUID isolatedSession = kernel.openSession(
                    true, 320, 240, URI.create("https://isolated.example/"));
            kernel.awaitIdle(isolatedSession).toCompletableFuture().get(3, TimeUnit.SECONDS);
            var isolatedTx = kernel.dispatch(isolatedSession, new BrowserCommand.SetSpatialMode(true))
                    .toCompletableFuture().get(2, TimeUnit.SECONDS);
            require(isolatedTx.parentRevision() == 1,
                    "new session inherited another session's revision");
            require(isolatedTx.committedRevision() == 2,
                    "isolated session revision chain is incorrect");

            require(Files.size(profile.resolve("journal.tsv")) > 0, "journal was not written");
            kernel.closeSession(isolatedSession);
            kernel.closeSession(session);
        }

        verifySupervisorLaunchFailureCleanup();
        System.out.println("KernelSelfTest: PASS");
    }

    private static void verifyViewportValidation(BrowserKernel kernel) {
        boolean rejected = false;
        try {
            kernel.openSession(false, 0, 420, URI.create("https://invalid.example/"));
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        require(rejected, "invalid viewport dimensions were accepted");
    }

    private static void verifySupervisorLaunchFailureCleanup() throws Exception {
        try (var supervisor = new ProcessSupervisor(ignored -> {})) {
            var spec = new ProcessSupervisor.ProcessSpec(
                    "missing-engine",
                    List.of("moagi-omega-command-that-does-not-exist"),
                    null,
                    Map.of(),
                    true,
                    1
            );
            expectLaunchFailure(supervisor, spec);
            require(supervisor.state(spec.id()) == null,
                    "failed native process launch left stale supervisor state");
            expectLaunchFailure(supervisor, spec);
        }
    }

    private static void expectLaunchFailure(
            ProcessSupervisor supervisor,
            ProcessSupervisor.ProcessSpec spec
    ) throws Exception {
        try {
            supervisor.start(spec);
            throw new AssertionError("missing process unexpectedly launched");
        } catch (IOException expected) {
            // Expected: the executable does not exist on any supported platform.
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
