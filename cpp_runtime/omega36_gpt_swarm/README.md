# Ω36 kinetic task swarm / Spatial Bytecode VM

Additive laboratory; does not replace the existing trillion3 kinetic hyperswarm or quine.
The logical lattice is (Z/10^12 Z)^3, with 10^36 addresses and 120-bit packed
indices. The finite C++ VM runs its canonical 64-bit ANN, encoder and decoder
programs; the Python adapter separately orchestrates a bounded active task graph.

## Execute
```sh
cd cpp_runtime/omega36_gpt_swarm
g++ -std=c++17 -O3 -Wall -Wextra -Werror moagi_omega36_svm.cpp -lcrypto -o svm
./svm selftest
./svm demo
./svm query 999999999999 0 1 8
python3 test_swarm.py --svm ./svm
```

## Kinetic law
For each active p, take a read-only snapshot of state and the six periodic
neighbors. A bounded thread pool dispatches one request per active node to a
caller-provided inference backend. Candidate outputs are accepted only when
an independent application score is a nonnegative integer strictly smaller
than the incumbent score. Otherwise the node value and memory are unchanged.
Accepted prior values enter bounded memory (last four by default).
When a VM path is supplied, serialize the complete candidate state, pad it
into a finite cubic tile, execute the actual C++ encode and decode bytecode,
and compare exact bytes before publishing the state. The SHA-256 receipt
identifies the canonical unpadded payload. Any archive failure aborts the
whole transaction. This is lossless byte encoding, not a semantic autoencoder.

The deterministic fixture decrements a nonnegative integer towards zero.
It tests scheduling, spatial adjacency, score descent, rollback, memory bounds,
and exact serialized reconstruction; it is not GPT or a general reasoning model.
The Python adapter does not implement the proposed 0x80–0x8E GPT opcodes in C++,
does not call a live model, and does not execute tools or generated code.
A caller can inject a real inference function and a task-specific validator;
the caller must enforce request deadlines, token/cost budgets, credential
handling, output safety and authorization. There is no claim of GPT weights,
semantic accuracy, or 10^36 physical nodes. The scheduler is in-process and
synchronous between rounds, not distributed. No network is used by the tests.

## Verification
CI runs the C++ selftest and the Python fixture tests (1,000 rounds of eight
nodes, a single separate cross-language archive transaction, error handling,
addressing and determinism). Actual measured receipts come from CI logs.
