package com.moagi.omega.core;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/** Supervises optional native engine host processes and reports exit/crash state. */
public final class ProcessSupervisor implements AutoCloseable {
    public record ProcessSpec(
            String id,
            List<String> command,
            Path workingDirectory,
            Map<String, String> environment,
            boolean restartOnFailure,
            int maximumRestarts
    ) {
        public ProcessSpec {
            id = Objects.requireNonNull(id, "id");
            command = List.copyOf(command);
            environment = Map.copyOf(environment);
            if (id.isBlank()) throw new IllegalArgumentException("Process id must not be blank");
            if (command.isEmpty()) throw new IllegalArgumentException("Process command must not be empty");
            if (maximumRestarts < 0) throw new IllegalArgumentException("maximumRestarts must be non-negative");
        }
    }

    public record ProcessState(
            String id,
            long pid,
            boolean alive,
            int restarts,
            Integer exitCode,
            Instant changedAt,
            String detail
    ) {}

    private final Map<String, Managed> managed = new ConcurrentHashMap<>();
    private final Map<String, CopyOnWriteArrayList<Consumer<ProcessState>>> listeners =
            new ConcurrentHashMap<>();
    private final Consumer<ProcessState> stateSink;
    private final WorkerPool workerPool = new WorkerPool();

    private static final class WorkerPool {
        final java.util.concurrent.ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    }

    public ProcessSupervisor(Consumer<ProcessState> stateSink) {
        this.stateSink = Objects.requireNonNull(stateSink);
    }

    /**
     * Registers a lifecycle listener for one supervised process. The current
     * state is replayed immediately when it already exists.
     */
    public AutoCloseable subscribe(String id, Consumer<ProcessState> listener) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(listener, "listener");
        var subscribers = listeners.computeIfAbsent(id, ignored -> new CopyOnWriteArrayList<>());
        subscribers.add(listener);
        ProcessState current = state(id);
        if (current != null) listener.accept(current);
        return () -> {
            subscribers.remove(listener);
            if (subscribers.isEmpty()) listeners.remove(id, subscribers);
        };
    }

    public synchronized void start(ProcessSpec spec) throws IOException {
        Objects.requireNonNull(spec, "spec");
        if (managed.containsKey(spec.id())) throw new IllegalStateException("Already supervised: " + spec.id());
        Managed entry = new Managed(spec);
        managed.put(spec.id(), entry);
        try {
            launch(entry);
        } catch (IOException | RuntimeException ex) {
            managed.remove(spec.id(), entry);
            entry.stopping = true;
            throw ex;
        }
    }

    public ProcessState state(String id) {
        Managed entry = managed.get(id);
        return entry == null ? null : entry.state;
    }

    public synchronized void stop(String id) {
        Managed entry = managed.remove(id);
        if (entry == null) return;

        entry.stopping = true;
        Process process = entry.process;
        if (process == null) return;

        process.destroy();
        try {
            if (!process.waitFor(2, TimeUnit.SECONDS)) process.destroyForcibly();
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            process.destroyForcibly();
        }
    }

    private void launch(Managed entry) throws IOException {
        if (entry.stopping || managed.get(entry.spec.id()) != entry) return;

        ProcessBuilder builder = new ProcessBuilder(new ArrayList<>(entry.spec.command()));
        if (entry.spec.workingDirectory() != null) builder.directory(entry.spec.workingDirectory().toFile());
        builder.environment().putAll(entry.spec.environment());
        builder.redirectErrorStream(true);

        Process process = builder.start();
        entry.process = process;
        entry.state = new ProcessState(entry.spec.id(), process.pid(), true,
                entry.restarts, null, Instant.now(), "started");
        publish(entry.state);

        workerPool.executor.submit(() -> drainOutput(process));
        workerPool.executor.submit(() -> monitor(entry, process));
    }

    private static void drainOutput(Process process) {
        try (var output = process.getInputStream()) {
            output.transferTo(OutputStream.nullOutputStream());
        } catch (IOException ignored) {
            // Process termination commonly closes the stream while the drain task is active.
        }
    }

    private void monitor(Managed entry, Process process) {
        try {
            int exit = process.waitFor();
            if (entry.process == process) {
                entry.state = new ProcessState(entry.spec.id(), process.pid(), false,
                        entry.restarts, exit, Instant.now(), "exited");
                publish(entry.state);
            }

            if (!entry.stopping
                    && managed.get(entry.spec.id()) == entry
                    && exit != 0
                    && entry.spec.restartOnFailure()
                    && entry.restarts < entry.spec.maximumRestarts()) {
                entry.restarts++;
                Thread.sleep(Math.min(5_000L, 250L * (1L << Math.min(entry.restarts, 4))));
                launch(entry);
            }
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        } catch (IOException ex) {
            entry.state = new ProcessState(entry.spec.id(), -1, false,
                    entry.restarts, null, Instant.now(), "restart failed: " + ex.getMessage());
            publish(entry.state);
        }
    }

    private void publish(ProcessState state) {
        stateSink.accept(state);
        for (Consumer<ProcessState> listener : listeners.getOrDefault(
                state.id(), new CopyOnWriteArrayList<>())) {
            try {
                listener.accept(state);
            } catch (RuntimeException ignored) {
                // A lifecycle observer cannot break process supervision.
            }
        }
    }

    @Override
    public void close() {
        for (String id : List.copyOf(managed.keySet())) stop(id);
        listeners.clear();
        workerPool.executor.close();
    }

    private static final class Managed {
        final ProcessSpec spec;
        volatile Process process;
        volatile ProcessState state;
        volatile int restarts;
        volatile boolean stopping;

        Managed(ProcessSpec spec) { this.spec = spec; }
    }
}
