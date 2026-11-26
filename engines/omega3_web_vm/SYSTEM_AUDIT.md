# Ω³ systems-wide integration and engineering gates

## Scope and operational invariants
The v2.0 baseline is a local Python/NumPy/SQLite reference VM, not a general LLM, autonomous code generator or production crawler. The 1000×1000 page plane with 1000 base token positions per page is a logical address space; continuation slabs extend it. Web and PDF use separate namespaces and weights. For page p and token offset i, M(p,i)=(p//1000,p%1000,i%1000,i//1000); inverse (1000x+y,1000s+z). Reversible bytes satisfy D_b(E_b(B))=B. Candidate neural states are accepted under reconstruction and integrity checks; rejected candidates preserve the prior model. No convergence theorem or claim of verified factual truth follows from these invariants.

## Evidence and limits
The archived summary and 1000 trace rows are from the local HTTP fixture run: four web snapshots, twelve chunks, 61,469 source bytes, 999 commits, one injected rollback (500), zero reconstruction bit errors, two retrieval probes. Initial reconstruction loss 0.00861643347889185, final 0.002664095489308238. These metrics concern training reconstruction on a small corpus, not general retrieval accuracy or live web search. Re-run independently with fresh output. Preserve source hashes, verify trace and do not treat timing or float hashes as cross-platform guarantees.

## Priorities
1. Adaptive lossless codec: compare raw, DEFLATE and XOR+DEFLATE; require exact round-trip and explicit format version. Existing codec is not uniformly smaller.
2. Retrieval: reject empty and unrelated queries, add calibrated abstention and held-out recall@K, MRR, nDCG. Current order-insensitive token features cannot distinguish reordered sentences; lexical shortlist caps recall. Add contextual embeddings and independent vector candidates, then fuse lexical/vector/graph ranks.
3. Shared multimodal memory: unify PDF/web addressing through typed namespaces and provenance. Preserve originals; distinguish fetched from imported content. Images/audio/video and generated answers remain unimplemented.
4. Scalability: cache versioned latents, shard indexes, incremental integrity manifests and bounded work queues. Benchmark 10k, 100k, 1M populated pages; virtual capacity is not throughput. Physical execution remains hardware-limited.
5. Actual spatial operators: explicit six-neighbour update and semantic edges, sparse active tiles, reproducible scheduling. The current coordinate map is not a parallel 3D processing topology.
6. Security: do not expose the current fetcher to untrusted users. DNS validation is not pinned to connections. Use isolated allowlisted egress, destination enforcement, total deadlines, rate and size bounds, robots/terms compliance; quarantine external data; do not execute restored HTML. Add access control, authenticated manifests, encrypted storage and recovery tests.
7. VM evolution: fixed opcode sequence today; add external instruction decoding, bounded registers/memory, verifier and sandboxed kernel candidates before any code evolution. Capability is not authority. No automatic deployment or merge.

## Reproduce
Run from this directory:
    python -m pip install -r requirements.txt
    python web_vm.py selftest --iterations 1000 --out /tmp/omega3-recheck
    python verify_trace.py /tmp/omega3-recheck
The archived trace can also be checked with python verify_trace.py test_output. Generated database, restored HTML and bytecode are intentionally not committed. This is an integration baseline and roadmap, not a claim that the proposed changes are implemented.
