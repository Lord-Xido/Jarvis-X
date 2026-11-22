# CloudBitFabric private deployment handoff (one-shot v6)

The public GitHub Actions gate at [run 45053151022](https://github.com/Lord-Xido/Jarvis-X/actions/runs/45053151022) **passed** on a hosted Ubuntu runner for synthetic P3D2 packet framing and TCP loopback (42,000 bytes, 42 fragments, two retransmissions, SHA-256 equality). The separate WAN echo check was **not verified**.

The complete CloudBitFabric v5 implementation is **not** committed to this public branch. The v6 operator bundle extends that private/local package with a fail-closed, one-command deployment and proof procedure.

On an existing Docker/Compose Linux server with DNS and port 80/443 configured, unpack the private v6 release package and run:

    python3 scripts/one_shot.py --mode deploy --domain your-real-hostname.co.za --report one_shot_evidence.json

The gate performs local regression tests, secret-file creation if needed, public-IP DNS validation, Compose preflight, image build/start, TLS 1.3 readiness/auth checks and real authenticated synthetic upload → compute → independently verified download → cleanup. It reports `REMOTE_HTTPS_PACKET_AND_COMPUTE_VERIFIED` **only after that path succeeds on the target server**. It does not purchase hosts or set DNS.

Dry run without Docker or cloud credentials:

    python3 scripts/one_shot.py --mode dry-run --report one_shot_evidence.json

That reports `LOCAL_VERIFIED_ONLY`, **not** worldwide transmission.

**Scope:** Keep v6 source, bearer secret, .env, database files, user payloads, deployment credentials and host access on the private server, not in this public repository. Production hardening remains required: tenant isolation, managed secret rotation, backups/restores, independent external verification, security review, bounded billing and durable distributed state before multi-replica scaling.
