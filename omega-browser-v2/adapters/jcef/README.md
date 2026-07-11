# JCEF adapter module

This directory reserves the Chromium compatibility adapter boundary.

The adapter must:

1. Implement `Engine.BrowserEngine` and `Engine.EngineSession`.
2. Keep CEF/JCEF classes outside the core module.
3. Translate CEF callbacks into `Engine.Event` records.
4. Enforce one browser-kernel session identity per engine session.
5. Forward permission requests to `CapabilityBroker`; never authorize directly.
6. Prefer off-screen/native surface delivery and explicit damage rectangles.
7. Report renderer termination through `Event.Crashed`.
8. Package and update native CEF binaries independently of the kernel JAR.

The source distribution does not bundle Chromium binaries or claim a working
JCEF integration. That integration requires platform-specific JCEF/CEF builds.
