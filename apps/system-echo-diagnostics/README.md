# Jarvis X System Echo Diagnostics

A browser-based HUD visualizer for the Jarvis-X project.

## What it shows

- simulated execution-stream telemetry and program-counter movement;
- synthetic CPU/GPU load bars;
- an echo-resonance waveform;
- a Three.js holographic icosahedral geometry core;
- adjustable recursion depth and rotational synchronization;
- a local browser clock.

## Run

Open `index.html` in a modern browser with network access to the Tailwind, Google Fonts, and Three.js CDNs.

## Operational boundary

This page is a **visual diagnostic simulation**. The bytecode stream, addresses, CPU/GPU percentages, and resonance bars are generated in the browser and are not measurements from the host operating system, CPU, GPU, microphone, network, or native Jarvis-X runtime.

The Three.js scene is real browser-side rendering; the telemetry surrounding it is synthetic UI state unless explicitly connected to a measured backend in a future implementation.
