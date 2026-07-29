# Security Policy

## Supported versions

Jarvis-X is alpha research software. Security fixes are applied to the default branch and the most recent tagged alpha release when practical.

| Version | Supported |
|---|---|
| `main` | Yes |
| latest `0.x` release | Best effort |
| older branches and unmerged PRs | No guarantee |

## Reporting a vulnerability

Use GitHub private vulnerability reporting for this repository when available.

If private reporting is unavailable, open a minimal issue stating that you have a potential security report and request a private communication channel. Do **not** include exploit code, secrets, personal data or details that would enable abuse in a public issue.

A useful report includes:

- affected commit, release or pull request;
- affected component;
- prerequisites and threat model;
- minimal reproduction;
- expected and observed behavior;
- realistic impact;
- suggested mitigation, if known.

Reports should receive an initial acknowledgement as repository availability permits. Complex research prototypes may require coordinated disclosure timelines.

## Security boundaries

### Bytecode execution

The Jarvis-X policy layer and cycle sandbox are not a complete hostile-code isolation boundary. Do not execute untrusted bytecode in a privileged process. Production isolation requires operating-system sandboxing, resource controls and a dedicated threat model.

### Persistent state

Ledger hashes detect unauthorized modification when verification is performed. They do not encrypt data, prove that the original input was truthful or prevent deletion of the entire ledger.

Do not journal secrets or sensitive personal information.

### Native code

C, C++, CUDA, OpenGL and browser-native experimental components may have memory-safety, driver and platform-specific risks. Build and run them in isolated development environments.

### Model files

Treat model checkpoints and repositories requiring `trust_remote_code=True` as executable code. Review source and provenance before loading them.

### Browser demonstrations

Browser interfaces may load external scripts or assets. Serve demonstrations from a controlled origin, review pinned dependencies and do not assume that client-side policy checks protect server-side resources.

### Self-optimization experiments

Bounded candidate search is not a security mechanism. Candidate evaluation must occur against isolated state, and rejected candidates must not retain authority or persistence side effects.

## Secrets

Never commit:

- API keys or tokens;
- private keys;
- cloud credentials;
- production database URLs;
- patient or personal records;
- private model or dataset access credentials.

Revoke and rotate any secret immediately if it is exposed, even if the commit is later removed.

## Dependency security

The repository uses automated dependency auditing and Dependabot configuration. A clean audit is a point-in-time signal, not proof of complete security.

## Disclosure

After remediation, the project may publish a concise advisory describing affected versions, impact, mitigation and credit, subject to the reporter's preferences and responsible-disclosure constraints.
