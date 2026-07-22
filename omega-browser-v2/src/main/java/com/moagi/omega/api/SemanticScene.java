package com.moagi.omega.api;

import com.moagi.omega.api.Geometry.Rect;
import com.moagi.omega.api.Geometry.Transform3D;

import java.net.URI;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class SemanticScene {
    private SemanticScene() {}

    public enum Action {
        FOCUS,
        CLICK,
        SET_VALUE,
        SCROLL_INTO_VIEW,
        COPY,
        OPEN_CONTEXT_MENU
    }

    public record Node(
            long id,
            long parentId,
            String role,
            String accessibleName,
            Rect bounds,
            Transform3D transform,
            Map<String, String> attributes,
            Set<Action> actions,
            List<Long> children,
            int layerDepth
    ) {
        public Node {
            role = Objects.requireNonNullElse(role, "generic");
            accessibleName = Objects.requireNonNullElse(accessibleName, "");
            bounds = Objects.requireNonNull(bounds, "bounds");
            transform = Objects.requireNonNullElse(transform, Transform3D.identity());
            attributes = Map.copyOf(Objects.requireNonNullElse(attributes, Map.of()));
            actions = Set.copyOf(Objects.requireNonNullElse(actions, Set.of()));
            children = List.copyOf(Objects.requireNonNullElse(children, List.of()));
        }
    }

    public record Snapshot(
            URI documentUri,
            Origin origin,
            long revision,
            long frameSequence,
            Instant capturedAt,
            List<Node> nodes
    ) {
        public Snapshot {
            Objects.requireNonNull(documentUri, "documentUri");
            Objects.requireNonNull(origin, "origin");
            Objects.requireNonNull(capturedAt, "capturedAt");
            nodes = List.copyOf(nodes);
        }

        public Node node(long id) {
            return nodes.stream()
                    .filter(node -> node.id() == id)
                    .findFirst()
                    .orElseThrow(() -> new IllegalArgumentException("Unknown semantic node: " + id));
        }
    }
}
