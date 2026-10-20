"""Export decoded RGB image, PCM waveform and MP4/GIF video."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import wavfile
import imageio.v2 as imageio


def rgb_uint8(image):
    # Shape C,H,W, values in [0, 1]
    return (np.clip(np.asarray(image).transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)


def save_media(prefix, decoded):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(decoded['image'][0].detach().cpu(), dtype=np.float32)
    audio = np.asarray(decoded['audio'][0].detach().cpu(), dtype=np.float32)
    video = np.asarray(decoded['video'][0].detach().cpu(), dtype=np.float32)
    png = str(prefix) + '.png'
    wav = str(prefix) + '.wav'
    mp4 = str(prefix) + '.mp4'
    gif = str(prefix) + '.gif'
    Image.fromarray(rgb_uint8(image), 'RGB').resize((256, 256), Image.Resampling.NEAREST).save(png)
    wavfile.write(wav, 1024, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    frames = [Image.fromarray(rgb_uint8(frame), 'RGB').resize((256, 256), Image.Resampling.NEAREST)
              for frame in video]
    for frame in frames:
        if frame.size != (256, 256):
            raise ValueError('unexpected video frame')
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=240, loop=0)
    try:
        # Real H264/MP4 video when an FFmpeg backend is installed.
        with imageio.get_writer(mp4, fps=4, codec='libx264', macro_block_size=1,
                                ffmpeg_log_level='error') as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        return {'image_png': png, 'audio_wav': wav, 'video_mp4': mp4, 'video_gif': gif}
    except (RuntimeError, OSError, ValueError, ImportError):
        Path(mp4).unlink(missing_ok=True)
        return {'image_png': png, 'audio_wav': wav, 'video_gif': gif, 'video_mp4': None}
