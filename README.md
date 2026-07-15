# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex
control layer and policy gate.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## Bounded inward optimisation

The repository includes an executable Level-1/2/3 optimisation controller
around a Level-0 execution evaluator. It proposes bounded hyperparameter,
architecture, and update-rule changes; evaluates each change in a shadow
state; rejects semantic, numerical, and resource violations; and atomically
commits only the best safe improvement.

The controller deliberately does not permit unrestricted source-code
self-modification. Active mechanics states are immutable and versioned, and
every candidate decision is journalled for inspection and rollback.

Run the deterministic demonstration:

```bash
python examples/inward_optimizer_demo.py
```

Run the tests:

```bash
pytest tests/test_inward_optimizer.py -v
```
