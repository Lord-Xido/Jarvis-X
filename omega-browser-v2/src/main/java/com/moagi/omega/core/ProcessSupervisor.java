package com.moagi.omega.core;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
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
            command = List.copyOf(command);
            environment = Map.copyOf(environment);
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
    private final Consumer<ProcessState> stateSink;
    private final WorkerPool workerPool = new WorkerPool();

    private static final class WorkerPool {
        final java.util.concurrent.ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    }

    public ProcessSupervisor(Consumer<ProcessState> stateSink) {
        this.stateSink = Objects.requireNonNull(stateSink);
    }

    public synchronized void start(ProcessSpec spec) throws IOException {
        if (managed.containsKey(spec.id())) throw new IllegalStateException("Already supervised: " + spec.id());
        Managed entry = new Managed(spec);
        managed.put(spec.id(), entry);
        launch(entry);
    }

    public ProcessState state(String id) {
        Managed entry = managed.get(id);
        return entry == null ? null : entry.state;
    }

    public synchronized void stop(String id) {
        Managed entry = managed.remove(id);
        if (entry != null && entry.process != null) {
            entry.stopping = true;
            entry.process.destroy();
            try {
                if (!entry.process.waitFor(2, java.util.concurrent.TimeUnit.SECONDS)) entry.process.destroyForcibly();
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
                entry.process.destroyForcibly();
            }
        }
    }

    private void launch(Managed entry) throws IOException {
        ProcessBuilder builder = new ProcessBuilder(new ArrayList<>(entry.spec.command()));
        if (entry.spec.workingDirectory() != null) builder.directory(entry.spec.workingDirectory().toFile());
        builder.environment().putAll(entry.spec.environment());
        builder.redirectErrorStream(true);
        entry.process = builder.start();
        entry.state = new ProcessState(entry.spec.id(), entry.process.pid(), true,
                entry.restarts, null, Instant.now(), "started");
        stateSink.accept(entry.state);

        workerPool.executor.submit(() -> monitor(entry));
    }

    private void monitor(Managed entry) {
        try {
            int exit = entry.process.waitFor();
            entry.state = new ProcessState(entry.spec.id(), entry.process.pid(), false,
                    entry.restarts, exit, Instant.now(), "exited");
            stateSink.accept(entry.state);

            if (!entry.stopping && exit != 0 && entry.spec.restartOnFailure()
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
            stateSink.accept(entry.state);
        }
    }

    @Override
    public void close() {
        for (String id : List.copyOf(managed.keySet())) stop(id);
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
