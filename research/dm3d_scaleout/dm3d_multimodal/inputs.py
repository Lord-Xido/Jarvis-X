"""Ingest real local image, WAV and video alongside tokenized text for inference.

The fixed vocabulary and spatial dimensions are a reference-implementation constraint.
"""
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample_poly
from .data import STOI, SEQ_LEN, IMAGE_SIDE, AUDIO_SAMPLES, FRAMES


def tokenize(text: str):
    words = re.findall(r"[a-zA-Z]+", text.lower())[:SEQ_LEN - 2]
    ids = [STOI['<bos>']] + [STOI.get(w, STOI['scene']) for w in words] + [STOI['<eos>']]
    return torch.tensor((ids + [STOI['<pad>']] * SEQ_LEN)[:SEQ_LEN], dtype=torch.long)


def convert_image(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        im = im.convert('RGB').resize((IMAGE_SIDE, IMAGE_SIDE), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(im).copy().astype(np.float32).transpose(2,0,1) / 255.0)


def convert_audio(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    rate, data = wavfile.read(path)
    if rate < 1 or rate > 192000:
        raise ValueError('unsupported source WAV sample rate')
    audio = np.asarray(data)
    if audio.ndim == 2:
        audio = audio.astype(np.float32).mean(axis=1)
    if audio.ndim != 1 or audio.size < 1:
        raise ValueError('audio must be a nonempty mono or multichannel WAV')
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32)
        audio = (audio - (float(info.min) + float(info.max)) / 2) / max(1., (float(info.max)-float(info.min))/2)
    else:
        audio = audio.astype(np.float32)
    import math
    divisor = math.gcd(int(rate),1024)
    resampled = resample_poly(audio, 1024//divisor, int(rate)//divisor)
    resampled = np.pad(resampled[:AUDIO_SAMPLES], (0,max(0,AUDIO_SAMPLES - len(resampled))))
    if not np.isfinite(resampled).all():
        raise ValueError('non-finite audio samples')
    return torch.from_numpy(np.clip(resampled, -1, 1).astype(np.float32))


def convert_video(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(path))
    frames = []
    try:
        for frame in reader:
            if len(frames) >= FRAMES:
                break
            im = Image.fromarray(frame).convert('RGB').resize((IMAGE_SIDE, IMAGE_SIDE), Image.Resampling.BILINEAR)
            frames.append(np.array(im, dtype=np.float32).transpose(2,0,1)/255.0)
    finally:
        reader.close()
    if not frames:
        raise ValueError('video has no decodable frames')
    frames.extend([frames[-1]] * (FRAMES - len(frames)))
    return torch.from_numpy(np.stack(frames).astype(np.float32))


def load_local(text: str, image_path: str, audio_path: str, video_path: str):
    return {'text': tokenize(text).unsqueeze(0),
            'image': convert_image(image_path).unsqueeze(0),
            'audio': convert_audio(audio_path).unsqueeze(0),
            'video': convert_video(video_path).unsqueeze(0)}
