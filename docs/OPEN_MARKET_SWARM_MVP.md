# Jarvis-X Open Source Open Market Swarm MVP

## Status

Integration candidate. This document describes the bounded open-market prototype implemented by `jarvisx.open_market` and `jarvisx.open_market_api`. It does not create a real payment network, public token, securities instrument or hostile-code execution service.

## Objective

The MVP turns the existing Jarvis-X transaction boundary into an open capability market:

```text
provider registration
    -> buyer task
    -> competitive plans/bids
    -> deterministic award
    -> SystemRuntime execution
    -> COMMIT / REJECT / FAIL
    -> verified execution receipt
    -> internal accounting settlement
    -> append-only market ledger
```

The governing invariant is:

```text
Plan != Execute != Commit != Settle
```

A provider can be economically credited only after the selected plan produces a committed Jarvis-X execution receipt.

## 1. Actors

### Buyer

Creates a bounded task with:

- required market capability;
- maximum accounting price;
- VM cycle and program-size budgets;
- candidate-count ceiling;
- utility weights for quality, cost, latency and risk.

### Provider

Advertises one or more market capabilities such as:

```text
optimization
rendering
simulation
scheduling
inspection
```

The registry is intentionally generic. A provider can later represent a human service, software agent, model endpoint, compute node, enterprise service or machine adapter.

### Jarvis-X market

The market validates participation, selects an admissible plan and delegates execution to the existing `SystemRuntime` / `CodexVM` boundary.

## 2. Task contract

For task `T_j`:

```text
T_j = (
  buyer,
  capability,
  max_price,
  resource_budget,
  granted_capabilities,
  utility_weights
)
```

The current reference implementation requires `vm.execute` and rejects bids whose program exceeds the buyer's declared word budget or whose requested capabilities exceed the buyer's grants.

## 3. Bid contract

Provider `i` submits:

```text
B_ij = (
  provider,
  program,
  price,
  quality,
  latency,
  risk,
  required_capabilities
)
```

A bid is rejected before award when:

- the provider does not exist;
- the task is not open;
- the provider does not advertise the requested capability;
- price exceeds the buyer ceiling;
- program length exceeds the task budget;
- required VM capabilities exceed the buyer's grants.

## 4. Deterministic market clearing

The prototype reuses the canonical deterministic planner:

```text
J(B_i) = w_q Q_i - w_c C_i - w_l L_i - w_r R_i
```

Only admissible bids enter the candidate set. The maximum `J` wins. Ties are resolved deterministically by bid id through the existing planner semantics.

This is a reference allocation function, not a claim that a real market should use one universal objective. Production deployments can define task-specific weights or replace the planner behind a compatible admission contract.

## 5. Verified execution

The winning bid is converted into an immutable `ExecutionRequest` and passed into `SystemRuntime`:

```text
winning program
    -> capability projection
    -> isolated CodexVM
    -> bounded execution
    -> VM ledger verification
    -> result hash
    -> system audit append
    -> COMMIT / REJECT / FAIL
```

The market does not directly mutate authoritative VM state.

## 6. Settlement

The MVP uses integer accounting units only.

For a successful committed execution:

```text
gross_units = winning_bid.price_units
platform_fee_units = floor(gross_units * fee_bps / 10000)
provider_units = gross_units - platform_fee_units
```

For rejected or failed execution:

```text
gross_units = 0
platform_fee_units = 0
provider_units = 0
```

This is accounting, not money movement. A future payment adapter can map verified settlement receipts to bank transfer, card, stable-value settlement, invoice, ERP payable or another legally appropriate rail without coupling payment authority into the VM kernel.

## 7. Provenance

The open market has its own `OmegaLedger` chain. Events include:

```text
provider.registered
task.created
bid.submitted
task.awarded
task.settled
task.failed
```

A settlement receipt binds:

- task and bid ids;
- buyer and provider ids;
- gross / fee / provider accounting units;
- execution status;
- execution request id;
- authoritative state hash;
- VM ledger head;
- system audit head;
- market ledger head.

The result is a three-level provenance chain:

```text
CodexVM ledger
    -> SystemRuntime audit receipt
    -> OpenMarket settlement ledger
```

## 8. API

Run the service during development with:

```bash
uvicorn jarvisx.open_market_api:app --reload
```

Primary endpoints:

```text
GET  /healthz
GET  /v1/capabilities
POST /v1/providers
GET  /v1/providers
POST /v1/tasks
GET  /v1/tasks/{task_id}
POST /v1/tasks/{task_id}/bids
GET  /v1/tasks/{task_id}/bids
POST /v1/tasks/{task_id}/award
POST /v1/tasks/{task_id}/execute
GET  /v1/tasks/{task_id}/settlement
GET  /v1/market
GET  /v1/ledger
```

### Example provider

```json
{
  "provider_id": "optimizer-a",
  "display_name": "Optimizer A",
  "capabilities": ["optimization"]
}
```

### Example task

```json
{
  "task_id": "factory-throughput-001",
  "buyer_id": "factory-a",
  "capability": "optimization",
  "max_price_units": 2000,
  "quality_weight": 100.0,
  "cost_weight": 0.01,
  "latency_weight": 1.0,
  "risk_weight": 10.0
}
```

### Example bid

```json
{
  "bid_id": "plan-a",
  "provider_id": "optimizer-a",
  "source": "SET A 42\nHALT",
  "price_units": 1200,
  "quality": 0.98,
  "latency": 1.0,
  "risk": 0.05
}
```

Execute with:

```text
POST /v1/tasks/factory-throughput-001/execute
```

A successful response includes both the computed state and the linked execution / market hashes.

## 9. Direct Python demonstration

```bash
python examples/open_market_demo.py
```

The example registers two optimization providers, creates one buyer task, submits two executable plans, awards the deterministic winner, executes through `SystemRuntime`, settles integer units and verifies the market ledger.

## 10. Security boundary

This MVP inherits the canonical Jarvis-X security boundary.

The current cycle sandbox is not hostile-code operating-system isolation. The public API must not be exposed as an unrestricted public bytecode-execution marketplace. Before production use with untrusted providers, add at minimum:

- authenticated identities;
- tenant and capability isolation;
- signed provider manifests;
- process/container isolation;
- CPU, memory, filesystem and network quotas;
- secrets isolation;
- rate limits and abuse controls;
- persistent transactional market storage;
- audit export and retention controls;
- reproducible provider package verification.

## 11. Commercial progression

The intended product progression is:

```text
open-source Jarvis-X kernel
    -> hosted verified-execution service
    -> enterprise private market
    -> industrial connectors
    -> third-party capability marketplace
    -> metered verified executions
```

The commercial unit can therefore become a Verified Execution (`VX`):

```text
VX = one authorized request
     + bounded execution
     + verification
     + committed receipt
```

The present MVP records the prerequisite evidence for that unit but intentionally does not implement billing, invoicing, tax handling, escrow or external settlement.

## 12. Next production gates

Before promotion to a canonical enterprise subsystem:

1. add durable SQL-backed market state and transactional recovery;
2. bind users/providers to authenticated identities and RBAC;
3. package provider plans as signed manifests rather than raw public assembly;
4. execute provider workloads inside hardened process/container isolation;
5. add task schemas and domain-specific verifiers;
6. add ERP/MES/PLM/IIoT connector contracts;
7. add usage metering and invoice/export adapters;
8. define SLA, dispute, retry and cancellation semantics;
9. benchmark market throughput and execution latency;
10. add machine-readable evidence artifacts to CI.
