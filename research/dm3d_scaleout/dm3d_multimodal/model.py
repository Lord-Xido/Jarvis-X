"""Trainable four-modality VAE, explicit 3D fusion and anchored inward refinement."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F
from .data import VOCAB, SEQ_LEN, IMAGE_SIDE, AUDIO_SAMPLES, FRAMES


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 512
    channels: int = 8
    grid: int = 4
    embed_dim: int = 24
    sequence_len: int = SEQ_LEN
    vocab_size: int = len(VOCAB)

    def __post_init__(self):
        if self.latent_dim < 4 or self.latent_dim > 2048:
            raise ValueError('latent_dim outside implemented budget')
        if self.channels != 8 or self.grid != 4 or self.embed_dim != 24:
            raise ValueError('current executable uses 8 channels, 4^3 lattice, d=24')
        if self.sequence_len != SEQ_LEN or self.vocab_size != len(VOCAB):
            raise ValueError('dataset vocabulary/sequence mismatch')


class DM3DMultiVAE(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.channels
        width = c * 4 * 4 * 4

        self.token_embedding = nn.Embedding(self.config.vocab_size, 24)
        self.token_position = nn.Parameter(torch.zeros(1, SEQ_LEN, 24))
        transformer = nn.TransformerEncoderLayer(
            d_model=24, nhead=4, dim_feedforward=48, dropout=0.0,
            batch_first=True, activation='gelu', norm_first=False)
        self.text_transformer = nn.TransformerEncoder(transformer, num_layers=1, enable_nested_tensor=False)
        self.text_project = nn.Sequential(nn.Linear(24, 96), nn.GELU(), nn.Linear(96, width))
        self.image_stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.GELU(), nn.Flatten(),
            nn.Linear(12 * 4 * 4, width))
        self.audio_stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(12 * 4 * 4, width))
        self.video_stem = nn.Sequential(
            nn.Conv3d(3, 8, 3, stride=(1, 2, 2), padding=1), nn.GELU(),
            nn.Conv3d(8, c, 3, stride=(1, 2, 2), padding=1), nn.GELU())
        self.modality_bias = nn.Parameter(torch.zeros(4, c, 4, 4, 4))
        # Cross-modal attention at every 3D grid cell (4 modality tokens / cell).
        self.cross_modal_attention = nn.MultiheadAttention(c, num_heads=2, batch_first=True)
        self.fuse = nn.Sequential(nn.Conv3d(c, 16, 3, padding=1), nn.GELU(),
                                  nn.Conv3d(16, c, 3, padding=1), nn.GELU())
        self.to_mu = nn.Linear(width, self.config.latent_dim)
        self.to_logvar = nn.Linear(width, self.config.latent_dim)
        self.from_latent = nn.Linear(self.config.latent_dim, width)
        self.decoder3d = nn.Sequential(nn.Conv3d(c, 16, 3, padding=1), nn.GELU(),
                                        nn.Conv3d(16, c, 3, padding=1), nn.GELU())
        self.text_head = nn.Linear(width, SEQ_LEN * self.config.vocab_size)
        self.image_head = nn.Linear(width, 3 * IMAGE_SIDE * IMAGE_SIDE)
        self.audio_head = nn.Linear(width, AUDIO_SAMPLES)
        self.video_head = nn.Linear(width, FRAMES * 3 * IMAGE_SIDE * IMAGE_SIDE)

    def encode(self, x: dict, availability=None):
        # Four different modality stems produce four spatially aligned 3D volumes.
        text = self.text_transformer(self.token_embedding(x['text']) + self.token_position)
        text = self.text_project(text.mean(dim=1))
        image = self.image_stem(x['image'])
        spectrogram = torch.stft(x['audio'], n_fft=32, hop_length=8,
                                 window=torch.hann_window(32, device=x['audio'].device),
                                 return_complex=True)
        audio = self.audio_stem(torch.log1p(spectrogram.abs()).unsqueeze(1))
        video = self.video_stem(x['video'].permute(0, 2, 1, 3, 4)).flatten(1)
        b = x['text'].shape[0]
        volumes = torch.stack([text, image, audio, video], 1).reshape(b, 4, 8, 4, 4, 4)
        volumes = volumes + self.modality_bias.unsqueeze(0)
        if availability is None:
            availability = torch.ones(b, 4, device=volumes.device, dtype=volumes.dtype)
        if availability.shape != (b, 4) or torch.any(availability.sum(dim=1) < 1):
            raise ValueError('availability must be [batch, 4] with at least one present modality')
        # Attention sequences correspond to one 3D voxel across all modalities.
        cells = volumes.permute(0, 3, 4, 5, 1, 2).reshape(b * 64, 4, 8)
        pad_mask = (availability <= 0).repeat_interleave(64, dim=0)
        attended, _ = self.cross_modal_attention(cells, cells, cells,
                                                  key_padding_mask=pad_mask,
                                                  need_weights=False)
        weights = availability.reshape(b, 1, 4, 1)
        fused_cells = (attended.reshape(b, 64, 4, 8) * weights).sum(dim=2) / weights.sum(dim=2)
        h0 = fused_cells.reshape(b, 4, 4, 4, 8).permute(0, 4, 1, 2, 3).contiguous()
        h = h0 + self.fuse(h0)
        mu = self.to_mu(h.flatten(1))
        logvar = self.to_logvar(h.flatten(1)).clamp(-8.0, 5.0)
        return mu, logvar, h

    def decode(self, z):
        b = z.size(0)
        h = self.from_latent(z).reshape(b, 8, 4, 4, 4)
        h = h + self.decoder3d(h)
        hidden = h.flatten(1)
        return {
            'text_logits': self.text_head(hidden).reshape(b, SEQ_LEN, self.config.vocab_size),
            'image': torch.sigmoid(self.image_head(hidden).reshape(b, 3, IMAGE_SIDE, IMAGE_SIDE)),
            'audio': torch.tanh(self.audio_head(hidden).reshape(b, AUDIO_SAMPLES)),
            'video': torch.sigmoid(self.video_head(hidden).reshape(b, FRAMES, 3, IMAGE_SIDE, IMAGE_SIDE)),
        }

    def forward(self, x, sample=True):
        mu, logvar, h = self.encode(x)
        z = mu + (torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else 0.0)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def refine(self, x, alpha=0.3, tolerance=0.002, max_steps=12):
        """Deterministic anchored encode/decode iteration. Not a convergence proof."""
        if not 0 < alpha < 1 or not 0 < tolerance < 1 or not 1 <= max_steps <= 64:
            raise ValueError('invalid inward refinement parameters')
        self.eval()
        anchor, _, _ = self.encode(x)
        z = anchor.clone()
        residuals = []
        for _ in range(max_steps):
            decoded = self.decode(z)
            reconstructed = {
                'text': decoded['text_logits'].argmax(-1),
                'image': decoded['image'],
                'audio': decoded['audio'],
                'video': decoded['video'],
            }
            rec_mu, _, _ = self.encode(reconstructed)
            next_z = (1 - alpha) * anchor + alpha * rec_mu
            residual = torch.linalg.vector_norm(next_z - z, dim=1).div(math.sqrt(z.size(1))).max().item()
            residuals.append(float(residual))
            z = next_z
            if residual <= tolerance:
                break
        return self.decode(z), z, {'residuals': residuals,
                                    'converged': bool(residuals[-1] <= tolerance),
                                    'steps': len(residuals), 'criterion': 'latent-RMS <= tolerance'}


def compute_losses(reconstruction, input_batch, mu, logvar, beta=0.0001):
    text = F.cross_entropy(reconstruction['text_logits'].transpose(1, 2), input_batch['text'])
    image = ((reconstruction['image'] - input_batch['image']) ** 2 * (1 + 3 * input_batch['image'])).mean()
    audio = F.mse_loss(reconstruction['audio'], input_batch['audio'])
    video = ((reconstruction['video'] - input_batch['video']) ** 2 * (1 + 3 * input_batch['video'])).mean()
    kl = 0.5 * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1))
    total = 0.8 * text + 2 * image + audio + 2 * video + beta * kl
    return {'total': total, 'text_ce': text, 'image_mse_weighted': image,
            'audio_mse': audio, 'video_mse_weighted': video, 'kl': kl}
