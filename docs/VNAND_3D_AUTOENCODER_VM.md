# 3D VNAND-Addressed ANN Autoencoder VM — C++17 reference

**Status:** software-only research simulator. This is not a physical NAND accelerator,
analog-MAC demonstration, calibrated energy model, or fabricated 4 GB die.

## Geometry and capacity

The depicted arrangement specifies 2 planes, 512 blocks per plane (1,024 total),
48 wordlines per block and 16,384 columns. Therefore:

    cells = 2 * 512 * 48 * 16384 = 805306368
    raw_tlc_bytes = cells * 3 / 8 = 301989888 = 288 MiB

The picture's 4 GB / 32 Gb capacity claim does not follow from these dimensions.
A 4 GiB TLC array would require at least 11,453,246,123 three-bit cells before
spares, ECC, or bad-block reserve. A real NAND density/floorplan must be supplied.

For visualization the logical coordinate is
(x,y,z) = (column % 128, column / 128, wordline);
plane and block are independent dimensions. This XY mapping is an indexing
choice, not a real die-floorplan claim.

## Dense autoencoder weight allocation

Logical weight address for neural row i and output neuron j:

    WL = first_WL + floor(i / rows_per_WL)
    column = (i mod rows_per_WL) * outputs + j
    block = 0

| Layer | Plane | Wordlines | Input rows per WL | Columns used per WL | Weights |
| --- | --- | --- | ---: | ---: | ---: |
| 784 → 256 | 0 | WL0–15 | 49 | 12,544 | 200,704 |
| 256 → 128 | 0 | WL16–23 | 32 | 4,096 | 32,768 |
| 128 → 256 | 1 | WL0–7 | 16 | 4,096 | 32,768 |
| 256 → 784 | 1 | WL8–23 | 16 | 12,544 | 200,704 |
| Total | | | | | 466,944 |

The built-in test checks all 466,944 addresses for uniqueness and bounds.
Only block zero of each plane is used; the host holds a sparse map and does
not allocate a physical 288 MiB NAND die.

Q3 cell state q is an integer in 0..7. Digital weight readback is

    w(q) = layer_scale * (q - 3)
    q(w) = clamp(round(w / layer_scale) + 3, 0, 7)

The eight states are ideal, not experimentally calibrated threshold-voltage
bands. Layer scales derive from a Xavier-like initial range.

## VM instruction stream

    LOAD_INPUT (784)
    MATVEC_Q3 layer0; RELU       -> 256
    MATVEC_Q3 layer1; RELU       -> 128
    TRANSFER128                 -> 128 logical inter-plane values
    MATVEC_Q3 layer2; RELU       -> 256
    MATVEC_Q3 layer3; SIGMOID    -> 784
    HALT

MATVEC_Q3 reads idealized quantized weights and uses **digital software**
multiplication and accumulation, not a flash-cell analog multiplier.
Conventional NAND uses series-connected string cells and selected-wordline
sensing. The proposed equation summing thresholded cell currents does not
establish a signed matrix dot product or a physically independent output
current per synapse.

The pre-existing Jarvis-X C++ 3D convolutional autoencoder is unchanged.
This feature is a distinct 28x28 dense Q3 VM.

## Gradients and abstract updates

    MSE = mean((X - reconstruction)^2)
    L = MSE + lambda * mean(abs(latent))

Backpropagation includes the MSE factor 2/784 and ReLU/sigmoid derivatives.
Quantization-aware training keeps float32 master weights as a surrogate
gradient state while inference and gradient propagation use quantized
readback weights. The discrete quantizer is not claimed to be differentiable.

An increased target level increments program_up_requests. A decreased level
increments erase_down_requests, and each affected plane/block increments a
block_rewrite_request at most once per training step. Because NAND erasure is
block-granular, independently erasing one weight is not a supported physical
operation. Shadow levels are changed immediately in software; there is
**no actual block erase**, restore, ISPP pulse, program-verify operation,
wear model, retention, ECC or physical sensor readout.

## Build and verify

    cmake -S cpp_runtime -B build/cpp-runtime -DJARVISX_BUILD_GL_VISUALIZER=OFF
    cmake --build build/cpp-runtime --target jarvisx-vnand-vm --parallel
    ctest --test-dir build/cpp-runtime --output-on-failure -R vnand-vm-reference
    ./build/cpp-runtime/jarvisx-vnand-vm --epochs 20

Self-test verifies address uniqueness, nominal TLC capacity, input validation,
deterministic model replay, valid 128-element latent transfer, output bounds
and a decreasing loss after 50 updates on one synthetic 28x28 square outline.
An optimized local GCC C++17 run reported MSE 0.251595 before and 0.174670
after the 50 training updates. This is a single-fixture result, not hardware
or generalization performance.

The reported 1.7 nJ inference energy and 2 mJ GPU comparison from the design
are **not substantiated**. The simple C * V^2 estimate omits array-wide loads,
sense/read sequencing, wordlines, peripherals, I/O, ADC, charge pumps and
training/program-erase energy. The 784/128=6.125 number is a dimension ratio,
not necessarily a byte compression ratio; bit widths and metadata matter.

## Hardware validation requirements

Provide a real die geometry and calibration data: Vth distributions,
voltage/current sensing, page selection timing, program-verify thresholds,
retention, disturb, endurance, block-rewrite overhead, ADC/control circuits,
and end-to-end latency/energy for specified workloads with a measured
CPU/GPU baseline.

## Literature

- https://www.mdpi.com/2073-431X/6/2/16 — NAND strings and cell programming.
- https://www.mdpi.com/2079-9292/10/12/1394 — TLC page buffer and ISPP.
- https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1279708/full — neuromorphic 3D NAND VMM designs and constraints.
