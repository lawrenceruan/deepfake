# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from importlib import import_module

_MODULES = {
    'flash_attention': '.attention', 'WanModel': '.model',
    'T5Decoder': '.t5', 'T5Encoder': '.t5', 'T5EncoderModel': '.t5', 'T5Model': '.t5',
    'HuggingfaceTokenizer': '.tokenizers', 'Wan2_1_VAE': '.vae2_1', 'Wan2_2_VAE': '.vae2_2',
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    'Wan2_1_VAE',
    'Wan2_2_VAE',
    'WanModel',
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
    'HuggingfaceTokenizer',
    'flash_attention',
]
