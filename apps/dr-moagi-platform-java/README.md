# Dr Moagi General Multimodal Platform — Java 17

A dependency-free Java 17 reference platform that evolves the original `DrMoagiEngine` into a capability-routed, recurrent multimodal computing kernel.

## What is operational

- **One recurrent kernel for seven surfaces:** chat, image, audio, video, code, data and compute.
- **4×4×4 toroidal latent field:** 64 state cells with periodic boundary conditions; boundary values fold back into the opposite face.
- **Turn-wise encoding:** each transcript turn contributes its own spatial feature field instead of repeatedly applying one whole-transcript feature vector.
- **Kinetic relaxation:** Euler integration updates the field toward the current encoded target with local neighbour coupling and a reconstruction-error force.
- **Authoritative telemetry:** loss is evaluated after the state update, so reported loss describes the state actually returned by the step.
- **Capability registry:** built-in adapters can be replaced by `ServiceLoader<DrMoagiPlatform.SurfaceAdapter>` implementations without modifying the kernel.
- **Offline compute:** `/compute` evaluates arithmetic with a safe parser; it does not execute shell commands or arbitrary code.
- **Offline data analytics:** `/data` computes count, sum, mean, min, max and population variance from numeric input.
- **Provider bridges:** an optional OpenAI-compatible text endpoint can drive chat/code; generic JSON HTTP endpoints can drive image/audio/video.
- **No embedded secrets:** provider configuration is read from environment variables.

## Mathematical kernel

For latent cell `i`, the periodic 3D neighbourhood mean is

```text
n_i = mean(s_(i±x), s_(i±y), s_(i±z))
```

The decoded local state and error are

```text
d_i = tanh(s_i + beta * (n_i - s_i))
e_i = z_i - d_i
```

and the explicit Euler update is

```text
F_i = alpha * (z_i - s_i)
    + beta  * (n_i - s_i)
    - gamma * s_i
    + eta   * e_i

s_i(t + dt) = clamp(s_i(t) + dt * F_i, -1, 1)
```

The step loss is evaluated on the **updated** field:

```text
L = mean_i (z_i - decode(s_next)_i)^2
```

This is a nonlinear recurrent state optimizer. It deliberately does **not** claim that fixed kernel parameters are being learned; trainable parameter optimisation can be added as a separate layer without conflating state relaxation and gradient learning.

## Build and run

```bash
cd apps/dr-moagi-platform-java
mkdir -p build
javac --release 17 -d build src/DrMoagiPlatform.java test/DrMoagiPlatformTest.java
java -cp build DrMoagiPlatformTest
java -cp build DrMoagiPlatform
```

Example session:

```text
/compute sqrt(81) + max(3,5) * 2^3
/data 10, 20, 30, 40
/image volumetric torus with inward flow
/code design a bounded work queue
/status
```

## Optional external text provider

```bash
export MOAGI_TEXT_ENDPOINT='https://provider.example/v1/chat/completions'
export MOAGI_API_KEY='...'
export MOAGI_MODEL='model-name'
java -cp build DrMoagiPlatform
```

The adapter uses an OpenAI-compatible **chat-completions-style** JSON contract. The platform itself does not depend on any vendor SDK.

## Optional media providers

Configure any subset:

```bash
export MOAGI_IMAGE_ENDPOINT='https://renderer.example/image'
export MOAGI_AUDIO_ENDPOINT='https://renderer.example/audio'
export MOAGI_VIDEO_ENDPOINT='https://renderer.example/video'
```

The platform posts this contract:

```json
{
  "mode": "image",
  "prompt": "...",
  "latent": [0.0, 0.0],
  "signature": "DM3D-..."
}
```

The endpoint may return any UTF-8 response body. Production media systems can replace this generic bridge with a provider-specific `SurfaceAdapter`.

## Extension contract

Implement:

```java
public final class MyImageAdapter implements DrMoagiPlatform.SurfaceAdapter {
  public DrMoagiPlatform.Surface surface() {
    return DrMoagiPlatform.Surface.IMAGE;
  }

  public String execute(DrMoagiPlatform.PlatformRequest request) {
    return "rendered artifact reference";
  }
}
```

Register the implementation using Java `ServiceLoader` metadata under `META-INF/services/DrMoagiPlatform$SurfaceAdapter`. Service-loaded adapters are registered after built-ins and therefore can override a built-in surface cleanly.

## Boundary of the reference implementation

This PR establishes the **platform kernel and adapter ABI**, not a claim that one local Java file itself contains production image synthesis, speech synthesis, video generation, a compiler, or a foundation model. Those capabilities are deliberately attached through adapters so they can be benchmarked, replaced and scaled independently.
