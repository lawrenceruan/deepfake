"""Real pretrained SAM encoder + detector forward/backward on synthetic inputs.

This is an implementation smoke test, not a trained-model accuracy evaluation.
It does not need Wan weights: synthetic grids exercise the cached-feature API.
"""
import argparse
import json
from pathlib import Path
import time
from contextlib import nullcontext
import torch
from deepfake.config import DetectorConfig
from deepfake.model import build_detector
from deepfake.losses import detector_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/wan_sam21_large_decoder_v1.json')
    p.add_argument('--sam2-checkpoint',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--frames',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new smoke report filename')
    if a.frames<1:p.error('frames must be positive')
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=DetectorConfig.load(a.config)
    start=time.monotonic()
    model=build_detector(None,cfg,a.sam2_checkpoint,a.device)
    model.heads.to(a.device);model.train()
    assert not model.rgb_encoder.training
    video=torch.rand(1,a.frames,3,256,256,device=a.device)
    frame=torch.ones(1,a.frames,dtype=torch.bool,device=a.device)
    latent_frames=1+(a.frames-1+3)//4
    grids=[torch.randn(1,cfg.backbone_dim,latent_frames,8,8,device=a.device) for _ in cfg.feature_layers]
    latent=torch.ones(1,latent_frames,dtype=torch.bool,device=a.device)
    amp=torch.autocast('cuda',dtype=torch.bfloat16) if torch.device(a.device).type=='cuda' else nullcontext()
    with amp:
        result=model.forward_heads(grids,latent,video,frame)
        target=torch.rand_like(result['modification_map'])
        losses=detector_loss(result,torch.ones(1,device=a.device),target,
                             torch.ones_like(target,dtype=torch.bool),frame)
    losses['loss'].backward()
    gradients=[float(adapter[0].weight.grad.abs().sum()) for adapter in model.heads.decoder.rgb_adapters]
    assert all(value>0 for value in gradients)
    assert all(p.grad is None for p in model.rgb_encoder.parameters())
    assert all(torch.isfinite(value).all() for value in result.values())
    report=dict(device=a.device,torch_version=str(torch.__version__),config=cfg.to_dict(),
        sam2_contract=model.rgb_encoder.contract,
        encoder_parameters=sum(p.numel() for p in model.rgb_encoder.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        map_shape=list(result['modification_map'].shape),loss=float(losses['loss'].detach()),
        adapter_gradient_sums=gradients,frozen_encoder_has_no_gradients=True,
        seconds=time.monotonic()-start,synthetic_inputs=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as handle:json.dump(report,handle,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
