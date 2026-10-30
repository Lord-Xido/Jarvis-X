# Dr Moagi Image ISA 3D — bounded exact reference

## Scope and coexistence

Module: src/jarvisx/image_isa3d.py

This implementation adds a separate, versioned .dmia2 format. The previously
demonstrated .dmia archive uses a model-bound Conv3D checkpoint; .dmia2 instead
uses a deterministic **non-neural occupancy latent** and an exact XOR residual.
It does not modify the existing 80K image ROM, RGB10 DM-PXBT codec, or their
serialization formats. There is no claim of learned image compression.

## Geometry

- 8K frame: 7680 x 4320 = 60 x 34 image tiles, each 128 x 128 RGB8.
- 1,000 image slots = 2,040,000 unique spatial tile addresses.
- Address = ((slot * rows) + tile_y) * columns + tile_x.
- One image tile has 49,152 bytes, divided into 96 bit cubes.
- Each bit cube has 16 x 16 x 16 = 4096 bit positions (512 bytes).
- Bitwise GF(2) transform T = delta_z o delta_y o delta_x.
- Inverse transform is the reverse prefix-XOR reconstruction.
- Each cube has 32 occupancy latent bytes, plus its exact 512-byte XOR residual.
- Algebra: residual = T(B) XOR estimate; B = inverse_T(estimate XOR residual).

## Container and security

The .dmia2 ZIP contains manifest.json, source.ppm, and coordinate-addressed
tiles. SHA-256 checks are performed on recovered source bytes, pixels, and
each individual tile. Missing/extra/duplicate archive members, unexpected
lengths, bad geometry, and altered residual data fail closed.

SHA-256 establishes integrity, not authentication. A malicious actor can
rewrite data and recalculate hashes without an external trusted signature.

Images currently require a binary P6 RGB8 PPM; JPEG/PNG decoding, trained
ANN inference, CUDA execution, cloud service, network transmission and
video synthesis are not provided by this module. Whole-frame in-memory
processing is bounded, not benchmarked at full 8K.

## CLI

    python -m jarvisx.image_isa3d encode input.ppm output.dmia2 --slot 42
    python -m jarvisx.image_isa3d inspect output.dmia2
    python -m jarvisx.image_isa3d decode output.dmia2 restored.ppm
    PYTHONPATH=src python -m pytest -q -o addopts='' tests/test_image_isa3d.py

## Verified locally

22 focused pytest cases passed: 12 random cube round trips, binary extremes,
edge tiles, exact PPM source and pixel recovery, address bounds, corruption
rejection and deterministic archives.

## Neural upgrade boundary

Add a separately versioned model-dependent codec with pinned checkpoint SHA,
a tested predictor interface, bounded residual accounting and held-out
accuracy/compression benchmarks. Do not change .dmia2 decode semantics or
treat neural reconstruction as lossless without the residual channel.
