package com.moagi.omega.api;

import java.time.Instant;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

/** Origin-bound, time-bound authority. */
public final class CapabilityToken {
    private final UUID tokenId;
    private final UUID sessionId;
    private final Origin origin;
    private final Capability capability;
    private final Instant expiresAt;
    private final boolean singleUse;
    private final AtomicBoolean consumed = new AtomicBoolean(false);

    public CapabilityToken(
            UUID tokenId,
            UUID sessionId,
            Origin origin,
            Capability capability,
            Instant expiresAt,
            boolean singleUse
    ) {
        this.tokenId = Objects.requireNonNull(tokenId);
        this.sessionId = Objects.requireNonNull(sessionId);
        this.origin = Objects.requireNonNull(origin);
        this.capability = Objects.requireNonNull(capability);
        this.expiresAt = Objects.requireNonNull(expiresAt);
        this.singleUse = singleUse;
    }

    public UUID tokenId() { return tokenId; }
    public UUID sessionId() { return sessionId; }
    public Origin origin() { return origin; }
    public Capability capability() { return capability; }
    public Instant expiresAt() { return expiresAt; }
    public boolean singleUse() { return singleUse; }
    public boolean consumed() { return consumed.get(); }

    public boolean validFor(UUID targetSession, Origin targetOrigin, Capability targetCapability) {
        return sessionId.equals(targetSession)
                && origin.equals(targetOrigin)
                && capability == targetCapability
                && Instant.now().isBefore(expiresAt)
                && !consumed.get();
    }

    public boolean consume() {
        return !singleUse || consumed.compareAndSet(false, true);
    }
}
