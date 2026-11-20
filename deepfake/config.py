from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class DetectorConfig:
    feature_layers: tuple = (9, 19, 29)
    backbone_dim: int = 3072
    hidden_dim: int = 256
    attention_heads: int = 8
    attention_blocks: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.0
    decoder_channels: int = 64
    probe_sigma: float = 0.0
    probe_seed: int = 1234
    num_train_timesteps: int = 1000
    map_loss_weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "feature_layers", tuple(self.feature_layers))
        if self.attention_heads < 1 or self.backbone_dim < 1 or self.mlp_ratio < 1 or self.num_train_timesteps < 1:
            raise ValueError("Head count, dimensions, ratio and timesteps must be positive")
        if not self.feature_layers or len(set(self.feature_layers)) != len(self.feature_layers):
            raise ValueError('Select unique feature layers')
        if any(i < 0 for i in self.feature_layers):
            raise ValueError('Layer indices must be nonnegative')
        if self.hidden_dim <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError('hidden_dim must be divisible by attention_heads')
        if self.attention_blocks < 1 or self.decoder_channels < 4:
            raise ValueError('Invalid attention/decoder size')
        if not 0 <= self.probe_sigma < 1:
            raise ValueError('probe_sigma must be in [0, 1)')
        if not 0 <= self.dropout < 1 or self.map_loss_weight < 0:
            raise ValueError('Invalid dropout/map loss weight')

    @classmethod
    def load(cls, path):
        with open(path) as f:
            obj = json.load(f)
        obj['feature_layers'] = tuple(obj.get('feature_layers', (9, 19, 29)))
        return cls(**obj)

    def to_dict(self):
        return asdict(self)
