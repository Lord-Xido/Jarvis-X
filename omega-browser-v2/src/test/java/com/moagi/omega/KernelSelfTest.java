package com.moagi.omega;

import com.moagi.omega.api.*;
import com.moagi.omega.core.*;
import com.moagi.omega.engine.mock.MockBrowserEngine;

import java.net.URI;
import java.nio.file.Files;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

public final class KernelSelfTest {
    public static void main(String[] args) throws Exception {
        var profile = Files.createTempDirectory("omega-kernel-test-");
        var selector = new EngineSelector();
        selector.register(new MockBrowserEngine());
        var broker = new CapabilityBroker((session, origin, capability, rationale) ->
                capability == Capability.GEOLOCATION
                        ? CapabilityBroker.Decision.ALLOW
                        : CapabilityBroker.Decision.DENY);

        try (var kernel = new BrowserKernel(selector, broker, profile)) {
            UUID session = kernel.openSession(false, 640, 420, URI.create("https://example.com/"));
            Thread.sleep(500);

            var snapshot = kernel.snapshot(session).toCompletableFuture().get(2, TimeUnit.SECONDS);
            require(snapshot.nodes().size() >= 5, "semantic snapshot missing nodes");
            require(snapshot.origin().scheme().equals("https"), "origin normalization failed");

            var tx = kernel.dispatch(session,
                    new BrowserCommand.SemanticAction(4, SemanticScene.Action.CLICK, java.util.Map.of()))
                    .toCompletableFuture().get(3, TimeUnit.SECONDS);
            require(tx.state() == TransactionJournal.State.COMMITTED, "semantic action not committed");

            var capabilityTx = kernel.dispatch(session,
                    new BrowserCommand.RequestCapability(Capability.GEOLOCATION, true))
                    .toCompletableFuture().get(2, TimeUnit.SECONDS);
            require(capabilityTx.state() == TransactionJournal.State.COMMITTED,
                    "capability transaction not committed");

            var rejected = kernel.dispatch(session,
                    new BrowserCommand.Navigate(URI.create("file:///etc/passwd")))
                    .toCompletableFuture().get(2, TimeUnit.SECONDS);
            require(rejected.state() == TransactionJournal.State.REJECTED,
                    "forbidden scheme was not rejected");

            require(Files.size(profile.resolve("journal.tsv")) > 0, "journal was not written");
            kernel.closeSession(session);
        }

        System.out.println("KernelSelfTest: PASS");
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
