# Security boundary

## Secrets

- No OpenAI API key is compiled into the launcher, PowerShell runtime, HTML, JavaScript,
  workflow, or repository history introduced by this subsystem.
- The browser UI does not receive the plaintext project key.
- API requests are issued by the loopback PowerShell service.
- The persisted key is encrypted with Windows DPAPI for the current user.

## Local service

- The HTTP listener binds to `127.0.0.1` only.
- A random launch token is required for the browser session.
- The application should not be exposed through port forwarding or a reverse proxy.

## Execution policy

- Model responses are treated as untrusted data.
- Model output is not executed as PowerShell, CMD, JavaScript, native code, or a system command.
- File attachments are forwarded only through declared media/API operations.

## Known limitations

- The launcher is unsigned.
- The backend is implemented in Windows PowerShell 5.1 for portability, not isolation strength.
- The browser UI is localhost-hosted rather than an embedded hardened WebView.
- Endpoint and model availability can change independently of this repository.
- The application has not been dynamically tested on every supported Windows build.

## Production hardening

Before production release:

1. code-sign the executable and release archive;
2. pin and review API/model configuration;
3. add Windows integration and failure-injection tests;
4. apply request-size, media-size, and concurrency quotas appropriate to the deployment;
5. add an explicit update channel with signed manifests;
6. conduct threat modelling for localhost token theft, browser extensions, and malicious files.
