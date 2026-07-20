#!/usr/bin/env python3
"""Export Jarvis-X weights in Hugging Face safetensors format.

Without --checkpoint, this creates deterministic initialized weights. They are
structurally valid but are not trained weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hf_model import JarvisXConfig, JarvisXModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="build/huggingface/JarvisX")
    parser.add_argument("--repo-id", default="LordXido/JarvisX")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--transition-layers", type=int, default=2)
    parser.add_argument("--omega-decay", type=float, default=0.95)
    parser.add_argument("--omega-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", help="Prefer HF_TOKEN or `hf auth login`.")
    parser.add_argument("--commit-message", default="Upload Jarvis-X safetensors checkpoint")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        payload = load_file(str(path), device="cpu")
    else:
        import torch

        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")

    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a state_dict mapping.")

    state_dict = {}
    for key, value in payload.items():
        if hasattr(value, "shape"):
            state_dict[key[7:] if key.startswith("module.") else key] = value
    if not state_dict:
        raise ValueError("No tensor weights were found in the checkpoint.")
    return state_dict


def make_model_card(repo_id: str, manifest: Dict[str, Any]) -> str:
    config = manifest["config"]
    return f'''---
license: mit
library_name: transformers
pipeline_tag: feature-extraction
tags:
- jarvis-x
- custom-code
- autoencoder
- state-space-model
- safetensors
---

# Jarvis-X

Jarvis-X is a custom `transformers` model implementing a bounded recurrent
auto-encoding state transition with prediction, residual correction, Ω-memory,
and a learned Λ projection gate.

## Checkpoint status

**{manifest["weight_status"]}**

- Parameters: `{manifest["parameter_count"]:,}`
- Seed: `{manifest["seed"]}`
- Input / hidden / latent / output: `{config["input_dim"]} / {config["hidden_dim"]} / {config["latent_dim"]} / {config["output_dim"]}`

## Load

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True).eval()
x = torch.randn(1, 8, model.config.input_dim)

with torch.inference_mode():
    output = model(input_values=x)

print(output.reconstruction.shape)
print(output.omega_state.shape)
```

Inspect the custom model code and pin a specific Hub revision in
security-sensitive deployments.
'''


def upload(output_dir: Path, args: argparse.Namespace) -> str:
    from huggingface_hub import HfApi, get_token

    token = args.token or os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Run `hf auth login` or set a write-enabled HF_TOKEN.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    return str(
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
            ignore_patterns=["__pycache__/*", "*.pyc"],
        )
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    config = JarvisXConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        output_dim=args.output_dim,
        transition_layers=args.transition_layers,
        omega_decay=args.omega_decay,
        omega_rate=args.omega_rate,
        architectures=["JarvisXModel"],
        auto_map={
            "AutoConfig": "configuration_jarvisx.JarvisXConfig",
            "AutoModel": "modeling_jarvisx.JarvisXModel",
        },
    )
    model = JarvisXModel(config).cpu().eval()

    if args.checkpoint:
        incompatible = model.load_state_dict(
            load_checkpoint(args.checkpoint), strict=args.strict_checkpoint
        )
        if not args.strict_checkpoint:
            if incompatible.missing_keys:
                print("Missing keys:", incompatible.missing_keys)
            if incompatible.unexpected_keys:
                print("Unexpected keys:", incompatible.unexpected_keys)
        weight_status = f"Imported checkpoint from `{args.checkpoint.name}`."
    else:
        weight_status = (
            "Deterministically initialized baseline; these weights are not trained. "
            "Train the model or pass --checkpoint before production use."
        )

    smoke_input = torch.zeros(2, 3, args.input_dim)
    with torch.inference_mode():
        smoke_output = model(input_values=smoke_input)
    expected_shape = (2, 3, args.output_dim)
    if tuple(smoke_output.reconstruction.shape) != expected_shape:
        raise RuntimeError("Model smoke test failed.")
    if not torch.isfinite(smoke_output.reconstruction).all():
        raise RuntimeError("Model produced non-finite values.")

    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    shutil.copy2(REPO_ROOT / "hf_model/configuration_jarvisx.py", output_dir)
    shutil.copy2(REPO_ROOT / "hf_model/modeling_jarvisx.py", output_dir)

    weight_files = sorted(output_dir.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError("No safetensors weights were produced.")

    manifest = {
        "format": "safetensors",
        "architecture": "JarvisXModel",
        "weight_status": weight_status,
        "seed": args.seed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "config": {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "output_dim": args.output_dim,
            "transition_layers": args.transition_layers,
            "omega_decay": args.omega_decay,
            "omega_rate": args.omega_rate,
        },
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weight_files
        ],
    }
    (output_dir / "weights_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        make_model_card(args.repo_id, manifest), encoding="utf-8"
    )

    from transformers import AutoModel

    loaded = AutoModel.from_pretrained(
        str(output_dir), trust_remote_code=True, local_files_only=True
    ).eval()
    with torch.inference_mode():
        reloaded = loaded(input_values=smoke_input)
    if tuple(reloaded.reconstruction.shape) != expected_shape:
        raise RuntimeError("AutoModel reload validation failed.")

    for cache_dir in output_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    print(json.dumps(manifest, indent=2))
    print(f"Built model: {output_dir}")
    if args.push:
        print(f"Uploaded: {upload(output_dir, args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
