# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from importlib import import_module

# Import only the requested pipeline. Feature extraction should not initialize
# speech/animation dependencies or query CUDA while importing a model class.
_EXPORTS = {
    'WanI2V': ('.image2video', 'WanI2V'),
    'WanS2V': ('.speech2video', 'WanS2V'),
    'WanT2V': ('.text2video', 'WanT2V'),
    'WanTI2V': ('.textimage2video', 'WanTI2V'),
    'WanAnimate': ('.animate', 'WanAnimate'),
}
__all__ = ['configs', 'distributed', 'modules', *_EXPORTS]


def __getattr__(name):
    if name in ('configs', 'distributed', 'modules'):
        value = import_module(f'.{name}', __name__)
    elif name in _EXPORTS:
        module, attr = _EXPORTS[name]
        value = getattr(import_module(module, __name__), attr)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
