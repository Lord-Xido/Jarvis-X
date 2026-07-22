package com.moagi.omega.api;

import java.net.URI;
import java.util.Locale;
import java.util.Objects;

/** Canonical security origin: scheme, host and effective port. */
public record Origin(String scheme, String host, int port) {
    public Origin {
        scheme = Objects.requireNonNullElse(scheme, "").toLowerCase(Locale.ROOT);
        host = Objects.requireNonNullElse(host, "").toLowerCase(Locale.ROOT);
    }

    public static Origin from(URI uri) {
        Objects.requireNonNull(uri, "uri");
        String scheme = Objects.requireNonNullElse(uri.getScheme(), "about");
        String host = Objects.requireNonNullElse(uri.getHost(), "");
        int port = uri.getPort();
        if (port < 0) {
            port = switch (scheme.toLowerCase(Locale.ROOT)) {
                case "https" -> 443;
                case "http" -> 80;
                default -> -1;
            };
        }
        return new Origin(scheme, host, port);
    }

    public boolean isSecureTransport() {
        return scheme.equals("https") || scheme.equals("about") || scheme.equals("mock");
    }

    public String serialize() {
        if (host.isBlank()) return scheme + ":";
        return scheme + "://" + host + (port > 0 ? ":" + port : "");
    }
}
