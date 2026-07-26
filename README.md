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

## Analytic SE(3) CUDA runtime

`cuda/se3/jarvis_x_se3.cu` is a standalone batched rigid-body exponential-map reference. One CUDA thread integrates one six-axis twist and emits one compact 3×4 pose. The executable includes pinned transfers, GPU-resident timing, an FP64 CPU oracle, geometric validation and separate end-to-end measurements.

```bash
nvcc -O3 -std=c++17 -lineinfo \
  cuda/se3/jarvis_x_se3.cu \
  -o jarvis_x_se3

./jarvis_x_se3 --count 1048576 --repeats 100 --dt 0.01
```

See [`cuda/se3/README.md`](cuda/se3/README.md) for the numerical, memory and benchmark contracts.
