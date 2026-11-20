"""Dataset-independent model-input contract for preprocessing adapters."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingConfig:
    schema_version: int = 1
    segment_duration_s: float = 3.0
    sampling_fps: float = 8.0
    size: int = 256
    temporal_padding: str = 'repeat_last_invalid'
    map_convention: str = 'rgb_mean_half_dssim_gaussian11_sigma1.5'
    endpoint_policy: str = 'adjacent_full_windows_share_endpoint'
    color_space: str = 'RGB'
    pixel_range: str = '0_1'

    def __post_init__(self):
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0
                   for v in (self.segment_duration_s,self.sampling_fps)):
            raise ValueError('Duration and sampling FPS must be finite and positive')
        steps=self.segment_duration_s*self.sampling_fps
        if not math.isclose(steps,round(steps)) or round(steps)%4:
            raise ValueError('duration * sampling_fps must be an integer multiple of four (Wan 4n+1)')
        if type(self.size) is not int or self.size<32 or self.size%32:
            raise ValueError('size must be a positive multiple of 32')
        supported=dict(schema_version=1,temporal_padding='repeat_last_invalid',
            map_convention='rgb_mean_half_dssim_gaussian11_sigma1.5',
            endpoint_policy='adjacent_full_windows_share_endpoint',color_space='RGB',pixel_range='0_1')
        for key,value in supported.items():
            if getattr(self,key)!=value:
                raise ValueError(f'Unsupported preprocessing setting {key}: {getattr(self,key)}')

    @property
    def maximum_frames(self):
        return int(round(self.segment_duration_s*self.sampling_fps))+1

    @property
    def fingerprint(self):
        canonical=json.dumps(asdict(self),sort_keys=True,separators=(',',':'))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text()))
