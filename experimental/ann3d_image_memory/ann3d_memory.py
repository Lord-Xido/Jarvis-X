"""Dr Moagi Ω³: real, bounded 3D CNN codec with quantised latent + exact residual.

Research demonstrator only. Depth from a single picture is SYNTHETIC, not scene depth.
No arbitrary file loading through pickle; .npz decoding uses allow_pickle=False.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn


class Autoencoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 8, 3, 2, 1), nn.ReLU(),
            nn.Conv3d(8, 12, 3, 2, 1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(12, 8, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose3d(8, 3, 4, 2, 1), nn.Sigmoid(),
        )


def make_volume(picture=None, width=96, height=64, depth=8):
    if depth % 4 or width % 4 or height % 4 or min(depth, height, width) < 4:
        raise ValueError('dimensions must be multiples of 4 and >= 4')
    if picture is None:
        y, x = np.mgrid[:height, :width]
        rgb = np.stack((255*x//max(width-1,1),
                        255*y//max(height-1,1),
                        (x*17+y*11)%256), axis=-1).astype(np.uint8)
    else:
        image = Image.open(picture).convert('RGB')
        rgb = np.asarray(ImageOps.fit(image, (width, height),
                          method=Image.Resampling.LANCZOS), dtype=np.uint8)
    # Synthetic shifted/brightened slices—not inferred physical depth.
    return np.stack([
        np.clip(np.roll(rgb, z-depth//2, axis=1).astype(np.float32)
                * (0.94+0.012*z), 0, 255).astype(np.uint8)
        for z in range(depth)], axis=0)


def _tensor(volume):
    return torch.from_numpy(volume.copy()).permute(3, 0, 1, 2).unsqueeze(0).float()/255.


def _decode(model, q, offset, scale):
    latent = torch.from_numpy(q.copy()).float() * (float(scale)/255.) + float(offset)
    with torch.no_grad():
        decoded = model.decoder(latent)
    return np.clip(np.rint(decoded[0].permute(1, 2, 3, 0).numpy()*255.), 0, 255).astype(np.uint8)


def encode(volume, archive, steps=160, seed=20261116):
    if volume.dtype != np.uint8 or volume.ndim != 4 or volume.shape[-1] != 3:
        raise ValueError('expect uint8 [depth,height,width,3]')
    depth, height, width, _ = volume.shape
    if any(d < 4 or d % 4 for d in (depth, height, width)):
        raise ValueError('volume dimensions must be multiples of 4 and >= 4')
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = Autoencoder3D()
    x = _tensor(volume)
    optim = torch.optim.Adam(model.parameters(), lr=0.018)
    for _ in range(steps):
        optim.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model.decoder(model.encoder(x)), x)
        loss.backward()
        optim.step()
    with torch.no_grad():
        latent = model.encoder(x)
    offset = np.float32(latent.min().item())
    scale = np.float32(max((latent.max()-latent.min()).item(), 1.e-8))
    q = np.clip(np.rint((latent.numpy()-offset)*255./scale), 0, 255).astype(np.uint8)
    approximation = _decode(model, q, offset, scale)
    residual = ((volume.astype(np.int16)-approximation.astype(np.int16)) % 256).astype(np.uint8)
    weights = {'model_'+k: v.detach().numpy() for k,v in model.state_dict().items()}
    destination = Path(archive)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, q=q, residual=residual,
                        offset=np.array(offset), scale=np.array(scale),
                        shape=np.array(volume.shape, dtype=np.int64),
                        sha256=np.frombuffer(hashlib.sha256(volume.tobytes()).digest(), dtype=np.uint8),
                        **weights)
    return approximation


def decode(archive, exact=True):
    # This prototype reads locally produced archives, not untrusted uploads.
    with np.load(archive, allow_pickle=False) as file:
        shape = tuple(int(v) for v in file['shape'])
        if len(shape) != 4 or shape[-1] != 3 or any(v < 4 or v % 4 for v in shape[:3]):
            raise ValueError('invalid 3D volume shape')
        if np.prod(shape, dtype=np.int64) > 12_000_000:
            raise ValueError('demo limits volume to 12M bytes')
        model = Autoencoder3D()
        expected = model.state_dict()
        state = {}
        for key, ref in expected.items():
            value = file['model_'+key]
            if value.shape != tuple(ref.shape):
                raise ValueError('invalid model weight shape')
            state[key] = torch.from_numpy(value.copy())
        model.load_state_dict(state)
        model.eval()
        q = file['q']
        residual = file['residual']
        if q.dtype != np.uint8 or tuple(q.shape) != (1, 12, shape[0]//4, shape[1]//4, shape[2]//4):
            raise ValueError('invalid latent tensor')
        if residual.dtype != np.uint8 or tuple(residual.shape) != shape:
            raise ValueError('invalid correction field')
        approximate = _decode(model, q, file['offset'], file['scale'])
        if exact:
            image = ((approximate.astype(np.uint16)+residual.astype(np.uint16)) % 256).astype(np.uint8)
            if not np.array_equal(np.frombuffer(hashlib.sha256(image.tobytes()).digest(), dtype=np.uint8), file['sha256']):
                raise ValueError('archive checksum mismatch')
            return image
        return approximate


def contact_sheet(source, approximate, reconstructed, path):
    z = source.shape[0]//2
    original = source[z]
    predicted = approximate[z]
    corrected = reconstructed[z]
    error = np.abs(original.astype(np.int16)-predicted.astype(np.int16)).mean(2)
    heat = np.clip(error*3, 0, 255).astype(np.uint8)
    heat_rgb = np.stack((heat, np.zeros_like(heat), 255-heat), axis=-1)
    images = [original, predicted, heat_rgb, corrected]
    labels = ['01  Source slice', '02  Neural decoder', '03  Error (scaled)', '04  Exact correction']
    w, h = original.shape[1], original.shape[0]
    scale = 4
    board = Image.new('RGB', (w*scale*2+48, h*scale*2+100), '#101f30')
    draw = ImageDraw.Draw(board)
    for idx, (label, im) in enumerate(zip(labels, images)):
        x0 = 16+(idx%2)*(w*scale+16)
        y0 = 12+(idx//2)*(h*scale+44)
        draw.text((x0,y0), label, fill='white')
        board.paste(Image.fromarray(im).resize((w*scale,h*scale), Image.Resampling.NEAREST), (x0,y0+25))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    board.save(path)


def demonstrate(args):
    volume = make_volume(args.input, args.width, args.height, args.depth)
    approximate = encode(volume, args.archive, steps=args.steps, seed=args.seed)
    reconstructed = decode(args.archive, exact=True)
    assert np.array_equal(reconstructed, volume)
    latent_only = 12*(args.depth//4)*(args.height//4)*(args.width//4)
    result = {
        'source_type': 'synthetic slices of one 2D image, not measured depth',
        'volume_shape': list(volume.shape), 'raw_bytes': int(volume.nbytes),
        'latent_bytes': latent_only,
        'latent_only_ratio': float(volume.nbytes/latent_only),
        'archive_bytes_including_model_and_residual': Path(args.archive).stat().st_size,
        'model_float32_bytes': int(sum(p.numel()*4 for p in Autoencoder3D().parameters())),
        'mse_before_residual': float(np.mean((approximate.astype(np.float32)-volume)**2)),
        'mismatched_channels_after_residual': int(np.count_nonzero(reconstructed!=volume)),
        'sha256': hashlib.sha256(volume.tobytes()).hexdigest(),
        'training_steps': args.steps
    }
    if args.preview:
        contact_sheet(volume, approximate, reconstructed, args.preview)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='Optional 2D RGB image (default: procedural test picture)')
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--archive', default='ann3d_memory.npz')
    parser.add_argument('--preview', default='ann3d_preview.png')
    args = parser.parse_args()
    demonstrate(args)


if __name__ == '__main__':
    main()
