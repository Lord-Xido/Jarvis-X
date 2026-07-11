package com.moagi.omega.core;

import com.moagi.omega.api.Capability;
import com.moagi.omega.api.CapabilityToken;
import com.moagi.omega.api.Origin;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/** Central authority broker. Engines never mint their own privileged access. */
public final class CapabilityBroker {
    public enum Decision { ALLOW, DENY }

    @FunctionalInterface
    public interface ApprovalHandler {
        Decision approve(UUID sessionId, Origin origin, Capability capability, String rationale);
    }

    private final ApprovalHandler approvalHandler;
    private final Map<UUID, CapabilityToken> tokens = new ConcurrentHashMap<>();

    public CapabilityBroker(ApprovalHandler approvalHandler) {
        this.approvalHandler = approvalHandler;
    }

    public Optional<CapabilityToken> request(
            UUID sessionId,
            Origin origin,
            Capability capability,
            boolean singleUse,
            String rationale
    ) {
        if (alwaysDenied(capability)) return Optional.empty();

        boolean automaticallyAllowed = capability == Capability.NETWORK
                && (origin.scheme().equals("https") || origin.scheme().equals("http")
                || origin.scheme().equals("mock") || origin.scheme().equals("about"));

        Decision decision = automaticallyAllowed
                ? Decision.ALLOW
                : approvalHandler.approve(sessionId, origin, capability, rationale);

        if (decision != Decision.ALLOW) return Optional.empty();

        Duration lifetime = singleUse ? Duration.ofMinutes(2) : Duration.ofMinutes(15);
        CapabilityToken token = new CapabilityToken(
                UUID.randomUUID(), sessionId, origin, capability,
                Instant.now().plus(lifetime), singleUse
        );
        tokens.put(token.tokenId(), token);
        return Optional.of(token);
    }

    public boolean validateAndConsume(
            UUID tokenId,
            UUID sessionId,
            Origin origin,
            Capability capability
    ) {
        CapabilityToken token = tokens.get(tokenId);
        if (token == null || !token.validFor(sessionId, origin, capability)) return false;
        boolean accepted = token.consume();
        if (accepted && token.singleUse()) tokens.remove(tokenId);
        return accepted;
    }

    public void revokeSession(UUID sessionId) {
        tokens.values().removeIf(token -> token.sessionId().equals(sessionId));
    }

    private static boolean alwaysDenied(Capability capability) {
        return capability == Capability.CREDENTIAL_ACCESS;
    }
}
