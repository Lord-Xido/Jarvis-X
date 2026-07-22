package com.moagi.omega.api;

import com.moagi.omega.api.SemanticScene.Action;

import java.net.URI;
import java.util.Map;
import java.util.Objects;

public sealed interface BrowserCommand permits
        BrowserCommand.Navigate,
        BrowserCommand.Reload,
        BrowserCommand.Stop,
        BrowserCommand.SemanticAction,
        BrowserCommand.RequestCapability,
        BrowserCommand.SetSpatialMode {

    record Navigate(URI uri) implements BrowserCommand {
        public Navigate { Objects.requireNonNull(uri, "uri"); }
    }

    record Reload() implements BrowserCommand {}
    record Stop() implements BrowserCommand {}

    record SemanticAction(long nodeId, Action action, Map<String, String> arguments)
            implements BrowserCommand {
        public SemanticAction {
            Objects.requireNonNull(action, "action");
            arguments = Map.copyOf(Objects.requireNonNullElse(arguments, Map.of()));
        }
    }

    record RequestCapability(Capability capability, boolean singleUse)
            implements BrowserCommand {
        public RequestCapability { Objects.requireNonNull(capability, "capability"); }
    }

    record SetSpatialMode(boolean enabled) implements BrowserCommand {}
}
