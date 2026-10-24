# DM3D v3 iterative DOCX bridge (public reference)

This bridge connects the public Jarvis-X repository to a **separately held trusted
local** Dr Moagi's Paper Computational System (DM-PCS) iterative Python SDK.
The private SDK, embedded DOCX, private bytecode, internal codec source, data,
and intellectual-property dossier **are not published or committed** here.

## Operational flow

    [trusted DOCX source]
          | inspect (bounded OOXML XML/ZIP)
          v
    [source SHA-256]
          | pinned local runtime (explicit human approval)
          v
    [VM oracle] -> [3D Morton tile encode] -> [nested 3D encode]
          ^                                        |
          |                                    [decode]
          |                                        |
    [fixed-point decision] <- [behavior and hash verification]
          |
          v
    [checkpoints + independent decode + 3D voxel receipt]

This is a **logical three-dimensional indexing system**. It does not imply
physical 3D GPU execution, model training, unbounded self-improvement, live
cloud execution, or complete Python semantic equivalence.

Let S_k be a candidate source and E_B / D_B the reversible block codec at
block size B. The private runtime tests D_B^L(E_B^L(S_k)) = S_k, checks
source SHA-256, runs a bounded Python-subset oracle, accepts only a strictly
better (VM steps, packed bytes, source bytes) lexicographic score when its
tested observable behavior is preserved, and stops on a fixed point or
the configured round budget.

The public bridge does **not** implement the private VM or certify arbitrary
program equivalence. It performs independent package decoding by invoking
the preapproved local runtime *again* and comparing reconstructed source bytes.
The logical Morton coordinates and counts are recorded in
public_bridge_proof.json. For very small sources, a 1x1x1 grid is expected.

## Quickstart

Inspect safely without executing code:

    python reference/dm3d_document_bridge/v3_iterative_adapter.py inspect /private/DM3D_DOCX_Recursive_Coding_Engine.docx

Run the verified local SDK after **personally reviewing** its source and
obtaining the SHA-256 from the actual local runtime script:

    sha256sum /private/DM3D_DOCX_ITERATIVE_SDK/dm3d_loop.py
    python reference/dm3d_document_bridge/v3_iterative_adapter.py run \
      /private/DM3D_DOCX_Recursive_Coding_Engine.docx \
      --runtime /private/DM3D_DOCX_ITERATIVE_SDK/dm3d_loop.py \
      --runtime-sha256 YOUR_REVIEWED_64_CHARACTER_SHA256 \
      --rounds 8 --layers 2 \
      --out /tmp/dm3d-v3-new-output \
      --approve-external-runtime

The output directory must be empty. The script never uploads a document or
its source to GitHub. It writes a local JSON receipt alongside the private
runtime's outputs (original.py, best.py, best.dm3d, decoded_verified.py,
iteration_report.json, snapshots/).

## Security boundaries

- Running the private runtime is **disabled** unless the explicit approval
  flag is supplied and the local script's SHA-256 matches the pinned digest.
- Pinning the entrypoint does **not** pin its imported dependencies; users
  must separately secure and audit the whole SDK directory/environment.
- A private runtime invoked in a subprocess is **not sandboxed** by this
  bridge. Only execute source and DOCX documents you trust.
- ZIP/XML size limits and checks detect some malformed inputs, not every
  OOXML security problem; do not treat inspection as a complete security audit.
- The public synthetic tests do not execute the private SDK on GitHub CI.
- Integrity digests are not encryption or cryptographic signatures.
- The private DOCX and SDK must remain outside this **public** repository.

## Tests and operational evidence

Run public tests in the isolated reference directory:

    cd reference/dm3d_document_bridge
    python -m unittest -q test_v3_iterative_adapter.py
    python -m unittest -q test_codec.py

On 2026-10-23 a separately held local 5-page DOCX and v2 iterative SDK
completed an end-to-end pinned run using this bridge. The local demo accepted
two source refinements, terminated at a fixed point, independently decoded
best.dm3d, matched the final SHA-256, and reported the outer 3D voxel address.
These are **local observations**, not GitHub CI results or proof of an
unrestricted programming engine. The private fixtures are not committed.
