# JARVIS X Phi3D · Native Mathematical Cloud VM v2

**Operational status**: reproducible, bounded 3D CPU-native application-level VM, separately integrated into `Lord-Xido/Jarvis-X` under `apps/chaotic-exploration-cloud-vm/`. There is no live Google Cloud deployment unless explicitly provisioned. This is **not** a hypervisor, pretrained ANN, GPU cluster, or automatic code-rewriting system. Read [MATHEMATICS.md](MATHEMATICS.md) for the exact recurrence, proof of spatial boundedness, mean-preservation invariant, and limitations.

## Components

- `vm.cpp` — C++17 fixed ISA: `INGEST 10 → LORENZ 20 → ATTEND_GRU 30 → FOLD_3D 40 → PLAN 50 → VERIFY 60 → HALT ff`. Ingest once; then a bounded recursion with RK4 Lorenz integration and scalar attention/gating, toroidal 3D folding, path search, and numerical audit.
- `server.py` — standard-library HTTP gateway. JSON input validation, cap of 32³ voxels and 100 iterations, 4 concurrent workers, 8-second native timeout, explicit voxels only when `N<=16`, optional bearer-like local control token header, CSP, no shell execution.
- `static/` — interactive panel rendering the **actual returned 3D field** as a reproducible isometric voxel projection, plus observed RMS trace. Can initialize a geometric sphere or sinusoidal field or a deterministic random lattice.
- `batch.py` — bounded run-to-completion command for Cloud Run Jobs and local batch execution; SHA-256 of canonical result evidence, not an authentication signature.
- `audit.py`, `tests/` — numerical invariants, deterministic replay, explicit input, invalid requests, and end-to-end HTTP integration.
- `Dockerfile` — multi-stage compiled non-root container; repo-scoped CI verifies CPU computation remotely.

## Run locally

Requires `g++` supporting C++17 and Python 3.10+; no Python packages required.

```bash
cd apps/chaotic-exploration-cloud-vm
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror vm.cpp -o jarvisx_vm
python3 -m unittest discover -s tests -v
python3 audit.py > audit.json
./jarvisx_vm --side 8 --steps 20 --emit-field 1
python3 batch.py
python3 server.py  # bind to 127.0.0.1:8080 by default
# Now visit http://127.0.0.1:8080
```

HTTP example with explicit 3D voxels:

```bash
python3 - <<'PY'
import json,urllib.request
n=4
payload={'side':n,'steps':5,'values':[0.0]*(n**3),'emit_field':True}
req=urllib.request.Request('http://127.0.0.1:8080/api/execute',
  data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
print(json.loads(urllib.request.urlopen(req).read()))
PY
```

Native stdin mode is also supported: `printf '0 %.0s' {1..64} | ./jarvisx_vm --side 4 --input-stdin 1 --emit-field 1 --chaos 0`. The spatial stability rule `6*alpha+beta<=1` is enforced in BOTH native and HTTP validation.

## Container

```bash
docker build -t jarvisx-phi3d .
docker run --rm --read-only --cap-drop=ALL --security-opt no-new-privileges \
  -p 127.0.0.1:8080:8080 jarvisx-phi3d
```

## Cloud Run: private container service (requires YOUR authorized Google Cloud project)

The following is a deployment **recipe**, not an action carried out during build or CI. Cloud Run is a managed container execution service, not independently provisioned hardware virtual machines. Verify quota, region, billing and IAM before deployment.

```bash
export PROJECT_ID='your-project-id'
export REGION='africa-south1'
export SERVICE='jarvisx-phi3d'
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
# Run from this application's directory, with Dockerfile as build root:
gcloud run deploy "$SERVICE" --source . --region "$REGION" \
  --no-allow-unauthenticated --cpu 1 --memory 512Mi --concurrency 4 \
  --min-instances 0 --max-instances 2 --timeout 30s
# Authenticate and test from a machine permitted to invoke the service:
gcloud run services proxy "$SERVICE" --region "$REGION" --port 8080
```

Cloud Run IAM authorizes access to the service **before** a request reaches the container. `JX_VM_CONTROL_TOKEN` is an optional extra application gate: pass `X-JX-Control-Token` with the secret and do not put secrets in browser source. For public deployments, add appropriate rate limits, monitoring, persistent audit storage, and application-level end-user authentication. The example is **private by default** and stores no user uploads.

## Cloud Run Jobs: separate run-to-completion mode

Google Cloud Run Jobs execute work to completion rather than serving HTTP. Deploy from source and override the container command; invoke on demand (costs may apply):

```bash
gcloud run jobs deploy jarvisx-phi3d-batch --source . --region "$REGION" \
  --command python3 --args batch.py --tasks 1 --parallelism 1 \
  --cpu 1 --memory 512Mi --max-retries 0 --task-timeout 60s
gcloud run jobs execute jarvisx-phi3d-batch --region "$REGION" --wait
```

Set `JX_JOB_REQUEST` on the job to a bounded JSON request as needed. For multiple independent seeded tasks, set `JX_JOB_SHARDED=1`: each task uses `seed + CLOUD_RUN_TASK_INDEX`, validated against the native range. This is parallel **independent** batch computation; it is not a connected distributed ANN, an object store, or a distributed convergence proof. No Cloud Run Jobs were created in a Google Cloud account here.

## Evidence boundaries

The original React artifact had virtual node counts and synthetic performance data; we do **not** copy or reuse its claims. Phi3D v2 reports native execution times and actual numerical state, explicitly flags unproven asymptotic convergence, and uses a non-cryptographic state fingerprint. Cloud CI verifies on GitHub-hosted Ubuntu CPU runners; those CI runs are real remote execution, **not** a live Cloud Run deployment. See [the original cloud VM v1 integration PR](https://github.com/Lord-Xido/Jarvis-X/pull/502) and its scoped checks for status.
