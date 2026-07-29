# Contributing to Jarvis-X

Jarvis-X welcomes focused contributions to deterministic virtual machines, bytecode formats, sparse spatial computation, numerical reference kernels, testing and documentation.

## Before opening code

1. Search existing issues and pull requests for overlapping work.
2. For a new architecture or large subsystem, open an issue first.
3. State whether the contribution is a canonical implementation, integration candidate, demonstration or specification.
4. Define the implemented boundary and the claims explicitly excluded.

## Development setup

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## Required local checks

Run the checks relevant to your change before opening a pull request:

```bash
python -m compileall -q src tests
flake8 src/jarvisx tests --select=E9,F63,F7,F82 --show-source --statistics
mypy src/jarvisx --ignore-missing-imports
pytest
```

Format changed Python files with:

```bash
black <changed-files>
isort <changed-files>
```

Native or browser subsystems must include their own build and validation commands.

## Pull-request scope

Prefer small, layered pull requests:

1. infrastructure or prerequisite repair;
2. core data contract;
3. implementation;
4. integration;
5. example or visualization.

Do not hide repository-wide repairs inside a large feature PR. Do not combine unrelated runtimes merely because they share terminology.

## Evidence requirements

### Correctness claims

Provide executable tests covering:

- normal behavior;
- invalid inputs;
- boundary values;
- deterministic replay where applicable;
- persistence, corruption and rollback behavior;
- dimensional or topology invariants.

### Performance claims

Provide:

- hardware and software environment;
- dataset and input sizes;
- warm-up and repetition protocol;
- baseline comparison;
- latency, throughput and memory together;
- raw or machine-readable results.

Virtual address-space size is not a performance result.

### Intelligence or adaptation claims

Define:

- the mutable variables;
- the objective function;
- the evaluation environment;
- the admissibility gate;
- rollback behavior;
- evidence that the metric cannot be trivially gamed.

Do not describe deterministic visualization, parameter search or online correction as consciousness or unrestricted self-evolution.

## Coding standards

- Use explicit types and names at public boundaries.
- Reject malformed inputs early.
- Keep persistent formats versioned and documented.
- Keep authoritative state separate from visualization and prediction state.
- Avoid hidden file, network or environment side effects.
- Inject clocks, random generators and external dependencies when deterministic tests need control.
- Prefer dependency-free reference kernels; isolate optional acceleration dependencies.
- Add comments for invariants and non-obvious trade-offs, not line-by-line narration.

## Commit messages

Use concise imperative messages, for example:

```text
Harden bytecode bounds validation
Add deterministic ledger replay fixture
Document sparse block persistence contract
```

## Documentation

Update the relevant documents when behavior changes:

- `README.md` for public usage;
- `docs/PROJECT_STATUS.md` for capability status;
- `docs/ARCHITECTURE.md` for architectural boundaries;
- `ROADMAP.md` for sequencing;
- `CHANGELOG.md` for user-visible changes;
- an ADR for material design decisions.

## Review checklist

A reviewer should be able to answer:

- What authoritative state changes?
- What is deterministic and what is environmental?
- What bounds memory and execution?
- What happens on malformed input or failure?
- How is the result validated?
- Which claim is implemented, and which remains proposed?

## Security

Do not publish secrets, private keys, personal medical information or production credentials. Report vulnerabilities according to [`SECURITY.md`](SECURITY.md).

## Conduct

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
