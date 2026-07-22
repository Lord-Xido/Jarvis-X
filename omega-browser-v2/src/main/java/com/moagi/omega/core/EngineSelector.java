package com.moagi.omega.core;

import com.moagi.omega.api.Engine.BrowserEngine;

import java.net.URI;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/** Policy-driven engine choice. More engines can be registered without changing the kernel. */
public final class EngineSelector {
    public record Selection(BrowserEngine engine, double score, String rationale) {}

    private final List<BrowserEngine> engines = new ArrayList<>();

    public synchronized void register(BrowserEngine engine) {
        engines.add(engine);
    }

    public synchronized Selection select(URI uri, boolean privateMode) {
        String scheme = uri.getScheme() == null ? "about" : uri.getScheme().toLowerCase(Locale.ROOT);
        return engines.stream()
                .filter(BrowserEngine::healthy)
                .filter(engine -> engine.capabilities().schemes().contains(scheme))
                .map(engine -> score(engine, uri, privateMode))
                .max(Comparator.comparingDouble(Selection::score))
                .orElseThrow(() -> new IllegalStateException("No healthy engine supports " + scheme));
    }

    public synchronized List<BrowserEngine> engines() {
        return List.copyOf(engines);
    }

    private Selection score(BrowserEngine engine, URI uri, boolean privateMode) {
        double score = 50;
        if (engine.capabilities().semanticSnapshots()) score += 20;
        if (engine.capabilities().siteIsolation()) score += 15;
        if (engine.capabilities().nativeSharedSurfaces()) score += 10;
        if (engine.capabilities().webGpu()) score += 5;
        if (privateMode && engine.capabilities().siteIsolation()) score += 5;
        return new Selection(engine, score, "capability-weighted automatic selection");
    }
}
