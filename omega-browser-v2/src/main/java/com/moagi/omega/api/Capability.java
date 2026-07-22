package com.moagi.omega.api;

/** Privileged operations that must be mediated by the browser kernel. */
public enum Capability {
    NETWORK,
    CLIPBOARD_READ,
    CLIPBOARD_WRITE,
    CAMERA,
    MICROPHONE,
    GEOLOCATION,
    FILE_READ,
    FILE_WRITE,
    CREDENTIAL_ACCESS,
    EXTERNAL_APPLICATION,
    NOTIFICATIONS
}
