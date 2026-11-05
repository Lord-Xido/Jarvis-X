"""Real gradient training/evaluation and safe checkpoint handling."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import DM3DMultiVAE, ModelConfig, compute_losses


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


def unpack(batch, device='cpu'):
    return {m: batch[m].to(device) for m in ('text', 'image', 'audio', 'video')}


def evaluate(model, dataset, batch_size=16):
    model.eval()
    totals = {}
    count = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            inp = unpack(batch)
            reconstruction, mu, logvar = model(inp, sample=False)
            losses = compute_losses(reconstruction, inp, mu, logvar)
            losses['token_accuracy'] = (reconstruction['text_logits'].argmax(-1) == inp['text']).float().mean()
            losses['image_mse'] = torch.nn.functional.mse_loss(reconstruction['image'], inp['image'])
            losses['video_mse'] = torch.nn.functional.mse_loss(reconstruction['video'], inp['video'])
            for key, val in losses.items():
                totals[key] = totals.get(key, 0.0) + float(val) * batch['text'].shape[0]
            count += batch['text'].shape[0]
    return {key: value / count for key, value in totals.items()}


def train(model, train_dataset, validation_dataset, epochs=8, batch_size=16, lr=0.003, seed=7, out_dir=None):
    if not (1 <= epochs <= 200 and 1 <= batch_size <= 128 and 0 < lr <= .1):
        raise ValueError('invalid training hyperparameters')
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    history = [{'epoch': 0, 'train_objective': None,
                'validation': evaluate(model, validation_dataset, batch_size)}]
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in loader:
            x = unpack(batch)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(x, sample=True)
            losses = compute_losses(reconstruction, x, mu, logvar)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            size = x['text'].shape[0]
            running += float(losses['total'].detach()) * size
            seen += size
        val = evaluate(model, validation_dataset, batch_size)
        history.append({'epoch': epoch, 'train_objective': running / seen, 'validation': val})
        print(f"epoch={epoch:02d} train={running / seen:.5f} val={val['total']:.5f} "
              f"textCE={val['text_ce']:.4f} image={val['image_mse_weighted']:.4f} "
              f"audio={val['audio_mse']:.4f} video={val['video_mse_weighted']:.4f}", flush=True)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'training_history.json').write_text(json.dumps(history, indent=2))
        save_checkpoint(model, out / 'dm3d_weights.pt')
    return history


def save_checkpoint(model, path):
    torch.save({'config': vars(model.config), 'state_dict': model.state_dict()}, str(path))


def load_checkpoint(path):
    payload = torch.load(str(path), map_location='cpu', weights_only=True)
    model = DM3DMultiVAE(ModelConfig(**payload['config']))
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model
