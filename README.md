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

## Dr Moagi M.M ROM Ω³ 6400³ reference runtime

The repository includes a sparse, bit-accurate reference implementation of the
3D auto-encoding and decoding engine. It models a logical 6400 × 6400 × 6400
lattice of 128-bit cells without allocating the full 4.194304 TB address space.

```python
from jarvisx.dr_moagi_3d import DrMoagiEngine, REQUIRED_VERIFICATION

engine = DrMoagiEngine()
decoded, committed, coordinate = engine.cycle(
    b"ABCDEF",
    REQUIRED_VERIFICATION,
    candidate_loss=1,
    active_loss=2,
)

assert decoded == b"ABCDEF"
assert committed
print(coordinate)
```

Run the focused tests with:

```bash
pip install -e ".[test]"
pytest -q tests/test_dr_moagi_3d.py
```

See `docs/DR_MOAGI_3D_BITWISE_RUNTIME.md` for the address equations, 128-bit
cell and instruction layouts, fixed-point datapath, Ω update and Λ commit gate.
