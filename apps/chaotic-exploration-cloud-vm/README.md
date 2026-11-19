# JARVIS X — Chaotic Exploration 3D Cloud VM

This is a **real, bounded, reproducible CPU-native, application-level 3D bytecode VM** with a standard-library HTTP gateway and web controls. The original uploaded React artifact is a browser-only visualization: its fake `deploy`/autoscale buttons and metric displays do not connect to a backend. The new dashboard reports measurements returned by the native binary, not the original emulation.

## Dataflow and supported operations

`INGEST (0x10) → LORENZ (0x20) → ATTEND_GRU (0x30) → FOLD_3D (0x40) → PLAN (0x50) → VERIFY (0x60) → HALT (0xff)`

* Fixed 3D toroidal lattice, side 4–32 and at most 32,768 resident voxels.
* Deterministic seeded initialization and Euler Lorenz dynamics.
* Scalar attention context with gated update. This **is not a pretrained attention network or trained GRU**.
* Six-neighbour periodic discrete Laplacian and mean contraction (bounded parameters).
* Four candidate paths with measured finite-horizon quadratic cost.
* Finite-state verification; RMS/max residuals and field energy measured each iteration; 64-bit **noncryptographic** FNV fingerprint; real process execution time.
* 1–100 iterations, with early stopping when residual RMS falls below threshold.

## Local test

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror vm.cpp -o jarvisx_vm
python -m unittest discover -s tests -v
python server.py
# open http://127.0.0.1:8080/
curl -sS -X POST http://127.0.0.1:8080/api/execute -H 'Content-Type: application/json' -d '{"side":8,"steps":6}'
```

## Container

```bash
docker build -t jarvisx-chaos-vm .
docker run --rm -p 127.0.0.1:8080:8080 --read-only --cap-drop=ALL --security-opt no-new-privileges jarvisx-chaos-vm
```

## Cloud Run — authenticated by default

Requires a Google Cloud project with billing, APIs enabled, `gcloud` logged in, and appropriate IAM. **These commands are not executed by committing the code.**

```bash
export PROJECT_ID='your-project-id'
export REGION='africa-south1'  # confirm regional availability for your project
export SERVICE='jarvisx-chaos-vm'
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy "$SERVICE" --source . --region "$REGION" \
  --no-allow-unauthenticated --cpu 1 --memory 512Mi \
  --concurrency 4 --max-instances 2 --timeout 30s
# Local authorized browser testing, when configured with IAM:
gcloud run services proxy "$SERVICE" --region "$REGION" --port 8080
```

Google Cloud Run hosts **container instances**, not independently provisioned hardware VMs. The platform controls autoscaling; the dashboard never invents instance counts. For persistent datasets and distributed multi-worker jobs, integrate Cloud Storage, Pub/Sub or Cloud Tasks, and audit logs with authenticated service accounts before claiming a distributed cluster. Production use also requires CSRF/session security if exposed to browsers beyond the IAM-protected proxy, quotas, monitoring, and resource controls.

**No live deployment, trained large-language-model backend, actual GPU, hypervisor VM or cryptographic integrity primitive is included.** The numerical program is an independently implemented CPU reference inspired by the attached prototype. The uploaded HTML contains external environment wrapper code and is deliberately not re-published.
