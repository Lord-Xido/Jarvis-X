# Architecture

## Process topology

```text
Windows GUI launcher
  -> extracts versioned assets to LocalAppData
  -> starts hidden Windows PowerShell runtime
  -> PowerShell binds loopback HTTP listener
  -> default browser opens tokenized localhost interface
  -> local backend performs authenticated OpenAI API requests
```

## Native launcher

`src/launcher.c` is a minimal PE32+ GUI executable with no C runtime dependency. The build
generates `embedded_assets.inc` from the HTML and PowerShell sources, then links only the
required Kernel32 and User32 imports declared in the `.def` files.

The launcher is intentionally not an AI execution environment. It only installs the bundled
assets and starts the local runtime.

## Trust boundary

```text
Untrusted browser input and model output
          |
          v
Loopback request validation and launch-token check
          |
          v
Declared API operation router
          |
          v
OpenAI HTTPS endpoints
```

The runtime does not expose a generic shell route. Chat output, generated media metadata,
and transcriptions remain data.

## Credential lifecycle

1. The user enters a project API key in the local settings interface.
2. The browser sends it once to the token-protected loopback backend.
3. The backend validates the key against the API.
4. The key is protected with current-user Windows DPAPI.
5. Subsequent requests decrypt it only inside the local backend process.

The repository and distributed executable contain no project credential.

## Media flows

- **Chat:** text and declared file/image inputs are normalized into an API request.
- **Image:** prompts are submitted to the configured image-generation model.
- **Speech:** text is submitted for audio synthesis; recorded audio is submitted for transcription.
- **Video:** the backend creates an asynchronous job, polls status, and downloads the result when complete.

## Build reproducibility

`lld-link /Brepro` derives a deterministic PE timestamp from the linked content. The CI job
builds twice from isolated copies and rejects the build unless `cmp` confirms byte identity.
Toolchain upgrades may legitimately change the executable hash; reproducibility is asserted
within one pinned CI environment and input revision.
