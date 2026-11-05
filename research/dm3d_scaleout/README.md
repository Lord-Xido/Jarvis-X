# Jarvis X / DM3D — Scale-Out Multimodal Engine

**Status:** working CPU reference with two-worker DDP and FSDP2 (parameter-sharded) smoke tests. **Not** a petabyte-scale benchmark, unrestricted multimodal foundation model, or production GPU deployment.

This project extends the existing PyTorch DM3D text/image/audio/video VAE with **bounded-memory data sharding, distributed training, distributed checkpoints, a disk-resident vector index, and lazy virtual 3D/2D tiling**. It preserves the 512-dimensional latent and anchored inward-loop methods in the original reference model.

## Operational pipeline

```text
JSONL / synthetic aligned text + image + audio + video
       ↓ (shard once, lazy per-worker loads)
Disk-backed .npy shards, size bounded by --shard-size
       ↓ DistributedSampler per torchrun rank
Distributed 3D VAE training
       ├─ DDP / gradient accumulation / CUDA autocast / rank-0 checkpoints
       └─ FSDP2 / sharded parameters / sharded DCP checkpoint
       ↓ optional full-model export for CPU retrieval
Per-record 512D latent encoder → immutable vector segments on disk
       ↓ bounded blockwise cosine scan
Top-k retrieval → anchored inward refinement → reconstruction, conditional and prior generation
```

## Requirements

- Python 3.10+; PyTorch 2.8+; NumPy, SciPy, Pillow, imageio (+ imageio-ffmpeg), pytest.
- CPU mode needs no GPU. GPU/multi-node DDP uses one rank per GPU and NCCL through PyTorch.
- Every training worker must see the same shard directory on a shared filesystem, or an identical local copy.
- Model input windows remain **8 token positions, 16×16 RGB, 256 audio samples, 4×16×16 video**, and 16 synthetic text vocabulary entries. Data sharding expands the number of records, **not the native media receptive field**. High resolution requires windowed processing and improved fusion across windows.

## Quickstart: make shards, train, index, query

```bash
python -m pip install -r requirements.txt
python -m dm3d_scale shard --output run --train-count 96 --validation-count 24 --shard-size 16
python -m dm3d_scale train --output run --epochs 2 --batch-size 4 --accumulation 2 --latent-dim 512
python -m dm3d_scale index --output run --segment-size 128
python -m dm3d_scale query --output run --sample-id 0 --top-k 5
python -m dm3d_scale generate --output run --split validation --sample-id 96 --top-k 3
python -m dm3d_scale footprint --height 8000000 --width 8000000 --tile 2048 --slots 122
```

## Data: scale beyond 4,096 rows

Provide a UTF-8 `.jsonl` file, **one record per line**:

```json
{"sample_id": 10, "split": "train", "text": "red circle moving tone", "image": "media/10.png", "audio": "media/10.wav", "video": "media/10.mp4"}
{"sample_id": 11, "split": "validation", "text": "blue square slow", "image": "media/11.png", "audio": "media/11.wav", "video": "media/11.mp4"}
```

Paths are interpreted relative to the JSONL file. The code reads only local files; no upload/network access occurs. It retains an O(number of records) byte-offset index while loading sample payloads lazily. All four modalities must be present in this v1 route; the model can separately encode availability-masked inputs when invoked directly.

```bash
python -m dm3d_scale shard --output real_run --manifest-jsonl /data/aligned.jsonl --shard-size 256
```

Manifest train/validation splits should be established by source identity to prevent leakage. Content must be licensed and handled in accordance with privacy obligations.

## Distributed data parallelism (DDP): many batches on many GPUs

Single node, 2 CPU processes (tested):

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 -m dm3d_scale train \
  --output run --epochs 2 --batch-size 3 --accumulation 2 --latent-dim 512 --amp none
```

Single node, 8 GPUs (launch template; **GPU results not tested in this package**):

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m dm3d_scale train \
  --output /shared/run --epochs 100 --batch-size 8 --accumulation 4 --latent-dim 512 --amp auto
```

Multinode template:

```bash
torchrun --nnodes=4 --nproc-per-node=8 --rdzv-id=dm3d-job-01 \
  --rdzv-backend=c10d --rdzv-endpoint=HOST:29400 \
  -m dm3d_scale train --output /shared/run --epochs 100 \
  --batch-size 8 --accumulation 4 --latent-dim 512
```

Each rank stores a full model replica; DDP increases batch throughput, but **does not shard parameters**. The code records rank-aggregated validation statistics without padding validation data. Train data uses a standard DistributedSampler, whose padding may repeat records to balance ranks. DDP resume currently requires the **same world size**. Checkpoints should be loaded only from trusted sources.

## FSDP2: model/optimizer state sharding

Two-process CPU sharding **tested** at 32D and 512D. This uses per-module `fully_shard` + PyTorch Distributed Checkpoint and produces separate, versioned checkpoint directories per completed epoch.

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m dm3d_scale train-fsdp2 --output run --epochs 3 \
  --batch-size 3 --latent-dim 512 --export-full
```

To resume training:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m dm3d_scale train-fsdp2 --output run --epochs 5 --batch-size 3 \
  --latent-dim 512 --resume --export-full
```

`--export-full` is optional and gathers a consolidated model on rank 0. **It requires enough host RAM for all model weights** and produces `checkpoint.pt` compatible with the index command. Ordinary FSDP2 checkpoints remain sharded. The cluster GPU/NCCL and rank-change/failure-recovery paths have **not** been qualified here.

## Virtual tiling and memory budget

For a logical 8,000,000 × 8,000,000 pixel image in RGBA FP32:

- Dense allocation = `8e6 × 8e6 × 4 × 4` = **1,024,000,000,000,000 bytes** (~1.024 PB decimal).
- A 2048 × 2048 RGBA FP32 tile = **67,108,864 bytes** (64 MiB).
- A 122-tile pool = **8,187,281,408 bytes** (7.625 GiB), excluding page tables, optimizer and other system allocations.
- Number of tile coordinates = `ceil(8e6/2048)^2` = **15,264,649**.

`dm3d_scale.tiles.windows` emits coordinate windows lazily; `crop_pad` and `stitch` support overlap-safe reconstruction for finite outputs. This **does not allocate** or process a full petabyte field; no remote GPU pool/NVMe scheduler is included in this package.

## Vector retrieval

`SegmentIndex` writes normalized 512D float32 embeddings and their original norms in immutable `.npy` segments and searches in bounded blocks. It returns **exact cosine nearest neighbors** under floating-point arithmetic, but search still requires O(N×d) work. The segment vectors live on disk and are memory mapped. Recovered unnormalized vectors support retrieval-conditioned decoding. The `generate` command additionally exports an unconditioned Gaussian prior sample; its semantic quality is not guaranteed. For millions/billions of embeddings with latency constraints, build an additional Faiss IVF/PQ or distributed ANN tier; this project **does not mislabel its scan as ANN**.

## Verified examples (CPU, not scalability benchmarks)

- `pytest -q tests`: **18 passed**, including the inherited DM3D baseline tests.
- DDP two CPU processes, 512D, 24 training / 8 validation scenes: epoch 2 held-out objective **3.210080**; resumed epoch 3 **2.889361**. These are synthetic-data demonstration results, not cross-dataset performance.
- FSDP2 two CPU processes, 512D, 24 training / 8 validation scenes: epoch 1 held-out objective **3.219956** and resumed epoch 2 held-out objective **2.045140**; full model exported, 24 embeddings indexed, and PNG/WAV/MP4/GIF generated after 2 anchored feedback iterations.
- FSDP2 32D two-worker checkpoint resume across epochs 1→2 passed.
- JSONL local-media route loaded a two-record train split and one-record validation split.

## Production gaps and next steps

1. Increase native image/audio/video resolution; replace fixed MLP output heads with chunkable convolutional or token decoding heads.
2. Out-of-core video/audio windowing and temporal/global cross-tile fusion.
3. Fault-tolerant manifests and atomic object-store checkpoint writes; GPU/NCCL validation with memory/throughput traces.
4. Faiss IVF/PQ, recall@k benchmarking, hierarchical multi-node query routing.
5. Evaluation on appropriately licensed, genuinely unseen multimodal data, with quality, coherence, drift and privacy metrics.

## Primary references

- PyTorch FSDP2 `fully_shard`: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- PyTorch torchrun: https://docs.pytorch.org/docs/stable/elastic/quickstart
- PyTorch distributed checkpoint: https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
- PyTorch fused attention: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- Faiss index selection: https://github.com/facebookresearch/faiss/wiki/guidelines-to-choose-an-index
- WebDataset large dataset layout: https://github.com/webdataset/webdataset
