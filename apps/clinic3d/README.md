# Clinic3D 0.3 — 3D bitwise staging VM

**WARNING: Synthetic-data engineering simulator only. Not a medical device, diagnosis,
triage system, guideline recommender or patient service. Never enter personal health
information. The keyword rules are intentionally unvalidated.**

This isolated folder integrates a front-end HTML/CSS/JS experience, an interpreter
for 14 fixed 32-bit VM instructions, 8x8x8 numerical lattice processing, reversible
3D byte-position shuffling, AES-256-GCM authenticated handoffs, a curated evidence
SOURCE index, and a server-authenticated admin pause control.

Run with Node.js >=20 (no npm dependencies):

    cd apps/clinic3d
    npm test
    npm run simulate
    npm run smoke
    npm start

Open http://127.0.0.1:8090. Server binds to loopback by default.

For administrator controls, set CLINIC_ADMIN_PASSWORD using a **private secret
manager**, minimum 16 characters, before running. Do not put secrets in source,
shell history, screenshots, or GitHub. The admin uses scrypt password verification,
HttpOnly SameSite=Strict cookie, CSRF check and a 15-minute session. This research
implementation keeps sessions, admin audit and kill-switch state only in RAM; it
does not satisfy production security or durability requirements.

The executable instruction order is:
INGEST -> ENCODE_3D -> PERCEIVE -> SEAL[0] -> OPEN[0] -> REASON ->
SEAL[1] -> OPEN[1] -> RETRIEVE -> SEAL[2] -> OPEN[2] -> GUARD -> EMIT -> HALT.

The signed-build mechanism from the earlier local v1 package is NOT claimed here:
the repository staging port enforces a fixed programme SHA-256 within the executable,
and its GitHub CI checks code behaviour. Strong release signing and independent
model/configuration verification are outstanding requirements.

Coordinates are mapped (x,y,z) -> (z+1,x+2,y+3) modulo cube side.
This is bijective obfuscation only; real confidentiality/integrity come from
AES-256-GCM with HKDF-derived stage keys, unique 96-bit nonces and bound
additional-authenticated-data (AAD) containing run ID, previous stage digest,
stage index, and the bytecode digest. Keys are random at process startup and
not committed to source. Intra-process encryption is NOT a service isolation
boundary or protection against server compromise.

GitHub Actions tests run on pull requests. GitHub is source-control deployment
only. No public cloud service, production secrets, clinical evidence ingestion,
clinician authorization or real-patient operation is deployed.

**4 Ps:** Product = synthetic 3D bitwise workflow; Price = free demonstrator,
no clinical billing; Place = GitHub PR + local private loopback staging; Promotion =
technical validation notes without claims of medical safety or efficacy.

Public health deployment requires licensed clinician oversight, clinically
validated multilingual assessment, guideline governance, independent safety
and security review, POPIA compliance and SAHPRA assessment where applicable.
