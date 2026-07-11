package com.moagi.omega.core;

import com.moagi.omega.api.BrowserCommand;
import com.moagi.omega.api.Origin;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

/** Append-only transaction journal used for audit and deterministic replay scaffolding. */
public final class TransactionJournal {
    public enum State { CREATED, VALIDATING, AUTHORIZED, EXECUTING, COMMITTED, REJECTED, FAILED, CANCELLED }

    public record TransactionSnapshot(
            UUID transactionId,
            UUID sessionId,
            long parentRevision,
            long committedRevision,
            Instant createdAt,
            Instant updatedAt,
            State state,
            String command,
            Origin origin,
            String detail
    ) {}

    private final Path file;
    private final List<TransactionSnapshot> entries = new ArrayList<>();

    public TransactionJournal(Path file) {
        this.file = file;
        try {
            Files.createDirectories(file.getParent());
        } catch (IOException ex) {
            throw new IllegalStateException("Cannot create journal directory", ex);
        }
    }

    public synchronized TransactionSnapshot create(
            UUID sessionId,
            long parentRevision,
            BrowserCommand command,
            Origin origin
    ) {
        Instant now = Instant.now();
        TransactionSnapshot snapshot = new TransactionSnapshot(
                UUID.randomUUID(), sessionId, parentRevision, -1, now, now,
                State.CREATED, describe(command), origin, "created"
        );
        append(snapshot);
        return snapshot;
    }

    public synchronized TransactionSnapshot transition(
            TransactionSnapshot previous,
            State state,
            long committedRevision,
            String detail
    ) {
        TransactionSnapshot next = new TransactionSnapshot(
                previous.transactionId(), previous.sessionId(), previous.parentRevision(),
                committedRevision, previous.createdAt(), Instant.now(), state,
                previous.command(), previous.origin(), detail == null ? "" : detail
        );
        append(next);
        return next;
    }

    public synchronized List<TransactionSnapshot> entries() {
        return List.copyOf(entries);
    }

    private void append(TransactionSnapshot snapshot) {
        entries.add(snapshot);
        String line = String.join("\t",
                snapshot.updatedAt().toString(),
                snapshot.transactionId().toString(),
                snapshot.sessionId().toString(),
                Long.toString(snapshot.parentRevision()),
                Long.toString(snapshot.committedRevision()),
                snapshot.state().name(),
                encode(snapshot.command()),
                encode(snapshot.origin().serialize()),
                encode(snapshot.detail())
        ) + System.lineSeparator();
        try {
            Files.writeString(file, line, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND, StandardOpenOption.WRITE);
        } catch (IOException ex) {
            throw new IllegalStateException("Cannot append transaction journal", ex);
        }
    }

    private static String describe(BrowserCommand command) {
        if (command instanceof BrowserCommand.Navigate navigate) return "NAVIGATE " + navigate.uri();
        if (command instanceof BrowserCommand.Reload) return "RELOAD";
        if (command instanceof BrowserCommand.Stop) return "STOP";
        if (command instanceof BrowserCommand.SemanticAction action) {
            return "SEMANTIC_ACTION node=" + action.nodeId() + " action=" + action.action();
        }
        if (command instanceof BrowserCommand.RequestCapability request) {
            return "REQUEST_CAPABILITY " + request.capability();
        }
        if (command instanceof BrowserCommand.SetSpatialMode mode) return "SPATIAL_MODE " + mode.enabled();
        return command.toString();
    }

    private static String encode(String value) {
        return Base64.getUrlEncoder().withoutPadding()
                .encodeToString(value.getBytes(StandardCharsets.UTF_8));
    }
}
