# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex control layer, policy gate, and a lossless 3D byte-ROM runtime for its native 64-bit instruction set.

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```

## Run assembly source

```bash
jarvisx run program.jx
```

Example program:

```text
SET A 7
SET B 5
ADD C A B
HALT
```

## Chrysalis-Ω 3D byte-ROM

Encode Jarvis-X source into a checksum-protected, fixed-address ROM image:

```bash
jarvisx rom encode program.jx program.jrom --engines 1 --grid 4x4x4
```

Inspect and verify the image:

```bash
jarvisx rom inspect program.jrom
jarvisx rom verify program.jrom
```

Execute the verified bytecode directly from the ROM image:

```bash
jarvisx rom run program.jrom
```

Create a bounded mutation candidate by changing only one `SET` immediate:

```bash
jarvisx rom mutate program.jrom candidate.jrom --word-index 0 --delta 5
```

The ROM runtime provides exact big-endian 64-bit bytecode round trips, engine-by-3D-cell addressing, SHA-256 integrity verification, capacity enforcement, atomic persistence, and bounded field-level mutation. It does not depend on a neural model or claim lossless reconstruction from an untrained latent vector.

See [the Chrysalis-Ω byte-ROM specification](docs/CHRYSALIS_OMEGA_BYTE_ROM.md) for the binary layout, equations, integrity rules, and operational guarantees.
