"""DM3D compact, trainable multimodal 3D autoencoder reference runtime."""
from .model import DM3DMultiVAE, ModelConfig
from .data import SceneDataset, VOCAB

__all__ = ['DM3DMultiVAE', 'ModelConfig', 'SceneDataset', 'VOCAB']
__version__ = '0.1.0'
