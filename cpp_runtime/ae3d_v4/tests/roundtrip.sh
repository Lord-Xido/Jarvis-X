#!/usr/bin/env bash
set -euo pipefail
bin="${1:?pass executable}"
source="${2:?pass source file}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# True quine: executable reproduces exact checked-in source bytes.
"$bin" --quine | cmp - "$source"

# Train on an actual three-dimensional U8 voxel volume.
"$bin" generate "$work/input.jxvol" 8 7
"$bin" train "$work/input.jxvol" "$work/model.jxmodel" 10 0.004

# Exact autoencoding: quantized latent + XOR residual.
"$bin" encode "$work/input.jxvol" "$work/model.jxmodel" "$work/exact.jxpkt" exact
"$bin" decode "$work/exact.jxpkt" "$work/model.jxmodel" "$work/restored.jxvol"
cmp "$work/input.jxvol" "$work/restored.jxvol"

# Lossy path and a direct world-coordinate query.
"$bin" encode "$work/input.jxvol" "$work/model.jxmodel" "$work/lossy.jxpkt" lossy
"$bin" decode "$work/lossy.jxpkt" "$work/model.jxmodel" "$work/lossy.jxvol"
"$bin" compare "$work/input.jxvol" "$work/lossy.jxvol"
"$bin" query "$work/exact.jxpkt" "$work/model.jxmodel" 1 2 3 | grep -q 'exact=1'

# Corruption must be rejected rather than decoded.
truncate -s -1 "$work/exact.jxpkt"
if "$bin" decode "$work/exact.jxpkt" "$work/model.jxmodel" "$work/invalid.jxvol" 2>/dev/null; then
  echo "ERROR: corrupted archive was accepted" >&2
  exit 1
fi
echo "PASS: quine, learned codec, exact bytes, lossy bytes, query and corruption rejection"
