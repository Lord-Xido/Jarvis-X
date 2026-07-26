# Jarvis-X

Jarvis-X is a deterministic, auditable virtual machine with a reflex control layer and policy gate.

## Architecture

The repository includes the canonical MM3D-AED-BCE-Ω⁴-G50T-OPT cosmogram: a three-dimensional auto-encoding, policy-projection, Z8 substrate-evolution, codebook-decoding, and chained SHA3-256 trace cycle.

- [MM3D-AED-BCE-Ω⁴ cosmogram specification](docs/MM3D_AED_BCE_OMEGA4_COSMOGRAM.md)
- Reference runtime: `src/jarvisx/mm3d_cosmogram.py`
- Determinism and ledger tests: `tests/test_mm3d_cosmogram.py`

The operational law is:

\[
X_{t+1}=D_{\mathcal C_t}\circ R_8\circ\Pi_{\Lambda_t}\circ E_{\Phi}(X_t)
\]

with chained provenance:

\[
\Omega_{t+1}=H(\Omega_t\|H(\operatorname{canon}(X_{t+1}))\|M_{t+1}).
\]

## Install

```bash
git clone https://github.com/Lord-Xido/Jarvis-X.git
cd Jarvis-X
pip install -r requirements.txt
pip install .
```
