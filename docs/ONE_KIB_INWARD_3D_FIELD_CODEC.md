# One-KiB Inward 3D Field Codec

## Status

This document defines the executable 1 KiB latent-field primitive used to connect the Jarvis-X
inward autoencoding architecture to a concrete virtual 3D world.

The implementation is in `src/jarvisx/latent_field_codec.py`.

The capability boundary is strict:

- the latent payload is exactly 1,024 bytes;
- the virtual address space is 1024 × 1024 × 1024 voxels;
- the decoder materializes only requested voxels, points, or slices;
- the codec is generative/lossy for arbitrary fields;
- it does **not** claim universal lossless compression of an arbitrary 1 GiB voxel volume into 1 KiB.

## 1. State geometry

The virtual world is

\[
X(x,y,z),\qquad 0\le x,y,z<1024.
\]

It contains

\[
1024^3=1,073,741,824
\]

addressable voxel locations.

The linear address is

\[
A=x+1024y+1024^2z,
\]

so the complete address space fits in 30 bits:

\[
0\le A<2^{30}.
\]

The persistent latent state is exactly

\[
z\in\{0,\ldots,255\}^{8\times8\times16}.
\]

Therefore

\[
8\times8\times16=1024\text{ bytes}.
\]

For an explicit unsigned 8-bit scalar world the logical expansion ratio is

\[
\frac{1024^3}{1024}=1,048,576:1.
\]

This ratio describes virtual addressability, not universal information-theoretic compression.

## 2. Encoder

The reference encoder samples a bounded scalar field at the 8×8×16 latent knot positions and
quantizes every sample to one byte:

\[
z_{ijk}
=
Q_8\left(
X\left(
\pi_x(i),
\pi_y(j),
\pi_z(k)
\right)
\right),
\]

where each \(\pi\) maps a latent coordinate onto the integer 1024-voxel world.

The reference implementation accepts a callable field:

```python
state = codec.encode_field(field)
```

and returns an immutable `LatentFieldState` whose payload is exactly 1,024 bytes.

A learned encoder can later replace the sampling transform without changing the packet contract.

## 3. Decoder

A voxel query does not allocate the billion-voxel world.

For a world coordinate \((x,y,z)\), the decoder maps the coordinate into continuous latent
coordinates:

\[
u=x\frac{7}{1023},\qquad
v=y\frac{7}{1023},\qquad
w=z\frac{15}{1023}.
\]

The decoded value is the trilinear interpolation of the eight neighboring latent bytes:

\[
\widehat X(x,y,z)
=
\sum_{a,b,c\in\{0,1\}}
\omega_{abc}(u,v,w)
\frac{z_{i+a,j+b,k+c}}{255}.
\]

Operationally:

```text
1 KiB state
   ↓
coordinate (x,y,z)
   ↓
find 8 latent neighbors
   ↓
compute interpolation weights
   ↓
decode one voxel
```

The virtual field is therefore addressable without being resident.

## 4. Lazy materialization

The API exposes three levels of decoding:

- `decode_voxel` — one coordinate;
- `decode_points` — a sparse coordinate batch;
- `materialize_slice` — a bounded 2D diagnostic slice.

A renderer or simulator should query only visible, active, colliding, or otherwise relevant
regions.

This is the runtime distinction:

\[
\text{virtual world size}\ne\text{resident decoded state}.
\]

## 5. Inward residual fold

Given observations

\[
\mathcal O=
\{(x_n,y_n,z_n,y_n^*)\}_{n=1}^{N},
\]

the current decoder predicts

\[
\widehat y_n=D(z,x_n,y_n,z_n).
\]

The weighted reconstruction loss is

\[
L(z)
=
\frac{
\sum_n w_n(\widehat y_n-y_n^*)^2
}{
\sum_n w_n
}.
\]

Because a trilinear query depends on exactly eight latent cells, its residual can be projected
directly back into those eight bytes:

\[
\frac{\partial L}{\partial z_j}
=
\sum_n
2w_n(\widehat y_n-y_n^*)\,
\omega_{nj}.
\]

The trial latent update is

\[
z^{trial}
=
Q_8\left[
\operatorname{clip}_{[0,1]}
\left(
\frac{z}{255}-\eta\nabla_zL
\right)
\right].
\]

This is the operational meaning of turning the field inward:

```text
world observation
      ↓
decoded prediction
      ↓
residual
      ↓
8-neighbor latent pullback
      ↓
1 KiB candidate state
      ↓
verify loss
      ↓
commit or rollback
```

The implementation performs bounded backtracking. A candidate is committed only when its
measured observation loss does not exceed the previous committed state.

## 6. Transactional recurrence

One refinement cycle is

\[
z_t
\rightarrow
D(z_t)
\rightarrow
R_t
\rightarrow
\nabla_zL_t
\rightarrow
z_{t+1}^{trial}
\rightarrow
\operatorname{VERIFY}
\rightarrow
\operatorname{COMMIT/ROLLBACK}.
\]

If accepted:

\[
L(z_{t+1})\le L(z_t).
\]

If every trial fails:

\[
z_{t+1}=z_t.
\]

The `revision` field advances only on a committed candidate.

## 7. Images, video, and live 3D animation

The same packet contract can be lifted to different coordinate meanings.

### Image

Treat one axis as a feature/depth axis or use a 2D specialization:

\[
I(x,y)\rightleftarrows z.
\]

### Video

Interpret the third coordinate as time:

\[
V(x,y,t)\rightleftarrows z.
\]

### Live 3D animation

Use space plus an external time/control parameter:

\[
S(x,y,z,t)=D(z,x,y,z,t).
\]

The current reference decoder is a static scalar-field primitive. A future learned decoder can add
time, vector channels, color, density, motion, material, topology, and neural-field parameters while
retaining the exact 1 KiB state budget.

## 8. Verification invariants

The executable reference enforces:

1. `len(payload) == 1024`;
2. coordinates remain inside the declared world;
3. field values are finite and in `[0,1]`;
4. the latent shape has exactly 1,024 cells;
5. refinement learning rates are finite and positive;
6. candidate bytes remain in `[0,255]`;
7. refinement is non-regressing on the supplied observation set;
8. decoding remains lazy unless a bounded slice is explicitly requested.

## 9. Minimal usage

```python
from jarvisx.latent_field_codec import FieldObservation, LatentFieldCodec

codec = LatentFieldCodec()

state = codec.encode_field(
    lambda x, y, z: (x + y + z) / (3 * 1023)
)

value = codec.decode_voxel(state, 512, 512, 512)

report = codec.refine(
    state,
    [
        FieldObservation(512, 512, 512, target=0.9),
        FieldObservation(128, 256, 384, target=0.2),
    ],
)

state = report.state
```

## 10. Extension path

The safe progression is:

```text
working scalar field
→ multi-channel field
→ sparse/octree query scheduler
→ learned encoder
→ learned coordinate-conditioned decoder
→ temporal latent dynamics
→ GPU batched decoder
→ live renderer
→ shadow verification
→ bounded promotion
```

This keeps the Jarvis-X rule intact: working first, then robust, portable, elegant, and advanced.
