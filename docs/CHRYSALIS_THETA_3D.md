# CHRYSALIS-Theta 3D Operational Runtime

## Scope

This module converts the CHRYSALIS-Theta arithmetic design into a dependency-free,
executable reference runtime. It is intentionally compatible with Python 3.8-3.12
and does not require PyTorch, CUDA, or pretrained encoders.

The runtime accepts modality embeddings, arbitrary numeric vectors, or text and
maps them into a coordinate-correct three-dimensional latent field.

## State

For grid dimensions `(W, H, D)`, the latent field is indexed as:

```text
X[z][y][x][channel]
```

with linear address:

```text
index = (z * H + y) * W + x
```

The operational state is:

```text
Theta_t = (X_t, G_t, I_t, E_t, A_t, H_t, C_t, Omega_t)
```

where:

- `X_t` is the projected 3D latent field;
- `G_t` is the locally smoothed routing-logit field;
- `I_t` is the exact top-k expert index set;
- `E_t` is the sparse expert response;
- `A_t` is multimodal cross-attention;
- `H_t` and `C_t` are recurrent hidden and cell fields;
- `Omega_t` is the deterministic state trajectory identified by its SHA-256 seal.

## Operational transition

```text
modalities
  -> deterministic modality encoding
  -> [D,H,W,d_model] grid projection
  -> local 6-neighbour routing field
  -> exact top-k selection
  -> coordinate-distinct low-rank expert execution
  -> weighted sparse reduction
  -> multi-head modality attention
  -> sparse ConvLSTM-like voxel update
  -> normalized output + deterministic state seal
```

The selected expert set is:

```text
I_k = TopK(g, k)
```

and only experts in `I_k` execute. This is true sparse dispatch rather than dense
convolution followed by small gate multipliers.

For selected expert `i`:

```text
u_i = ReLU(A_i x_i)
y_i = tanh(x_i + B_i u_i)
y   = sum_i softmax(g_i) * y_i
```

The reference implementation derives coordinate-specific low-rank matrices from a
deterministic seed. This gives independent expert arithmetic without allocating a
dense 512 x 512 matrix for every voxel.

## Recurrent 3D memory

Selected voxels update a local ConvLSTM-like state using their six spatial
neighbours. Unselected voxels decay by `state_decay`:

```text
i_t = sigmoid(a_x x_t + a_h h_{t-1} + a_n neighbour + b_i)
f_t = sigmoid(b_x x_t + b_h h_{t-1} + b_n neighbour + b_f)
o_t = sigmoid(c_x x_t + c_h h_{t-1} + c_n neighbour)
c_t = f_t * c_{t-1} + i_t * candidate * gate_weight
h_t = o_t * tanh(c_t)
```

The complete recurrent state contains:

```text
2 * W * H * D * d_model
```

floating-point values.

## CLI

```bash
chrysalis-theta @input.json --summary-only
```

Example input:

```json
{
  "config": {
    "width": 4,
    "height": 4,
    "depth": 4,
    "d_model": 512,
    "top_k": 2,
    "expert_rank": 8,
    "num_heads": 8,
    "seed": 3297095957
  },
  "sequence": [
    {
      "text": "motion through a three-dimensional field",
      "image": [0.1, 0.4, -0.2, 0.7],
      "audio": [0.2, -0.1, 0.5],
      "video": [0.0, 0.2, 0.4, 0.6]
    }
  ]
}
```

A numeric vector of length `d_model` is treated as an existing embedding. Other
numeric vectors and text are deterministically feature-hashed into `d_model`.
Production CNN, transformer, audio, and video encoders can therefore be connected
as upstream adapters without changing the 3D core.

## Arithmetic diagnostics

Every result reports:

- total grid positions;
- exact number of active experts;
- activation ratio `k / P`;
- approximate scalar operation counts for projection, routing, experts,
  attention, and recurrence;
- recurrent-state storage;
- selected 3D coordinates and normalized gate weights;
- deterministic state hash.

These counts describe the reference arithmetic actually executed. They are not
hardware throughput or latency claims.

## Guarantees and boundaries

The tests enforce:

- non-cubic coordinate/index round trips;
- exact top-k cardinality;
- normalized sparse gate weights;
- deterministic output and state hashes;
- recurrent evolution and reset replay;
- coordinate-distinct expert behaviour;
- attention tensor shape `heads x modalities`;
- fail-closed configuration validation;
- CLI sequence execution.

This module is an operational arithmetic reference, not a trained foundation
model. The deterministic feature hashing and seeded experts are replaceable by
learned encoders, routers, and expert parameter banks. It does not claim measured
A100/TPU latency, universal approximation for the concrete finite configuration,
or guaranteed out-of-distribution generalization.
