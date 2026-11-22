# PDM-3D-Ω P3D2 cloud-route compatibility gate

This directory contains a **public-safe, synthetic-data protocol probe**, not the complete private/local CloudBitFabric v5 application or its production receiver.

## Evidence boundaries

* The probe constructs the exact v2/v3/v5 P3D2 **102-byte** header, 12-byte padded 90-bit 3D coordinate, CRC32 and 16-byte truncated HMAC-SHA-256 using an ephemeral test key.
* It sends 42 packets encoding 42,000 synthetic bytes over **real loopback TCP**, injects one lost fragment and one corrupted fragment, retries and checks exact SHA-256 reconstruction.
* The optional WAN probe sends only a small synthetic JSON test marker through public **HTTPS** POST echo endpoints and checks the returned marker. It **does not send a full PDM3D packet** to a public receiver.
* The GitHub Actions worker is a **temporary cloud-hosted runner**; it is not a continuously running cloud service and does not expose an externally reachable receiver.
* This code was compared byte-for-byte against the local v5 frame encoder using a fixed UUID and synthetic payload before publication. That interop claim is a **local source comparison**, not a remote full-v5 execution claim.
* The PDM3D v5 local bundle, proprietary source, cloud credentials and tenant data have **not** been published by this test.

## Run

    python3 experimental/pdm3d_cloud_route/cloud_route_probe.py --report feedback.json
    python3 experimental/pdm3d_cloud_route/cloud_route_probe.py --wan-probe --report feedback.json

Workflow: .github/workflows/pdm3d-cloud-route.yml. Download the pdm3d-cloud-route-feedback artifact to review runner-side SHA-256, route scope and optional WAN status.

## To reach actual production cloud

Provision a user-controlled HTTPS hostname and bounded VM/container; deploy the separately vetted v5 package with secrets outside source control; configure monitoring, backups and identity; then perform a real sender-to-receiver transfer from an independent network. The GitHub Actions gate is an intermediate cloud verification step, **not a substitute**.
