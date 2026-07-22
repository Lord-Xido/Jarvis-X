package com.moagi.omega.api;

import com.moagi.omega.api.Engine.Event;
import com.moagi.omega.core.TransactionJournal.TransactionSnapshot;

import java.util.UUID;

public sealed interface BrowserEvent permits
        BrowserEvent.EngineEvent,
        BrowserEvent.TransactionChanged,
        BrowserEvent.CapabilityChanged,
        BrowserEvent.SessionChanged {

    record EngineEvent(UUID sessionId, Event event) implements BrowserEvent {}
    record TransactionChanged(UUID sessionId, TransactionSnapshot transaction) implements BrowserEvent {}
    record CapabilityChanged(UUID sessionId, CapabilityToken token, String decision) implements BrowserEvent {}
    record SessionChanged(UUID sessionId, String state, String engineId) implements BrowserEvent {}
}
