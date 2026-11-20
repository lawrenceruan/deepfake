"""No checkpoint downloads: test RGB geometry, gradients and baseline isolation."""
from dataclasses import replace
import json
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from deepfake.config import DetectorConfig
from deepfake.model import DetectorHeads, WanOnlyDetector, WanRGBDetector, restore_detector
from deepfake.sam2_rgb import SAM2RGBEncoder, load_image_encoder_state
from deepfake.train import trial_forward
from deepfake.losses import detector_loss

torch.set_num_threads(2)


class TinyImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3,256,1)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return {'backbone_fpn': [self.projection(F.avg_pool2d(x,s)) for s in (4,8,16)]}


def config():
    return DetectorConfig(feature_layers=(0,1,2), backbone_dim=12, hidden_dim=16,
        attention_heads=4, decoder_channels=8, rgb_branch='sam2.1_hiera_large_decoder',
        sam2_image_size=64, sam2_frame_batch_size=2)


def inputs():
    grids = [torch.randn(2,12,2,2,2) for _ in range(3)]
    latent = torch.tensor([[True,True],[True,False]])
    frames = torch.tensor([[True]*5,[True]+[False]*4])
    rgb = [torch.randn(2,5,256,32//s,32//s) for s in (4,8,16)]
    return grids,latent,frames,rgb


def test_encoder_normalization_chunking_freeze_and_padding():
    encoder = TinyImageEncoder()
    branch = SAM2RGBEncoder(encoder,64,2).train()
    assert not branch.training and not encoder.training
    video=torch.full((2,3,3,32,32),.5)
    frame=torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool)
    video[1,1:]=float('nan')
    outputs=branch(video,frame)
    assert [len(x) for x in encoder.seen]==[2,2]
    torch.testing.assert_close(encoder.seen[0][0,:,0,0],
        (torch.tensor([.5,.5,.5])-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225]))
    assert [tuple(x.shape) for x in outputs]==[(2,3,256,8,8),(2,3,256,4,4),(2,3,256,2,2)]
    assert all(not x.requires_grad and not x[1,1:].any() for x in outputs)
    single=branch(video[1:2,:1],frame[1:2,:1])
    for batched,one in zip(outputs,single):torch.testing.assert_close(batched[1:2,:1],one)


def test_rgb_changes_maps_but_not_classification_and_padding_is_ignored():
    torch.manual_seed(7)
    heads=DetectorHeads(config()).eval()
    grids,latent,frame,rgb=inputs()
    a=heads(grids,latent,(5,32,32),frame,rgb)
    b=heads(grids,latent,(5,32,32),frame,[x+torch.randn_like(x) for x in rgb])
    torch.testing.assert_close(a['visual_logit'],b['visual_logit'],rtol=0,atol=0)
    assert not torch.allclose(a['modification_map'][0],b['modification_map'][0])
    changed=[x.clone() for x in rgb]
    for x in changed:x[1,1:]=float('nan')
    c=heads(grids,latent,(5,32,32),frame,changed)
    torch.testing.assert_close(a['modification_map'],c['modification_map'])
    assert not c['modification_map'][1,1:].any()
    one=heads([g[1:2,:,:1] for g in grids],latent[1:2,:1],(1,32,32),frame[1:2,:1],
              [x[1:2,:1] for x in rgb])
    torch.testing.assert_close(a['modification_map'][1:2,:1],one['modification_map'])


def test_map_gradient_reaches_adapters_and_wan_but_not_frozen_encoder():
    branch=SAM2RGBEncoder(TinyImageEncoder(),64,2)
    model=WanRGBDetector(None,branch,config()).train()
    grids,latent,frame,_=inputs()
    video=torch.rand(2,5,3,32,32)
    result=model.forward_heads(grids,latent,video,frame)
    target=torch.rand_like(result['modification_map'])
    losses=detector_loss(result,torch.tensor([0.,1.]),target,torch.ones_like(target,dtype=torch.bool),frame)
    losses['classification'].backward(retain_graph=True)
    assert all(p.grad is None for p in model.heads.decoder.rgb_adapters.parameters())
    model.zero_grad()
    losses['map'].backward()
    assert all(adapter[0].weight.grad.abs().sum()>0 for adapter in model.heads.decoder.rgb_adapters)
    assert model.heads.blocks[0].temporal.in_proj_weight.grad.abs().sum()>0
    assert all(p.grad is None for p in branch.parameters())


def test_optional_finetuning_and_checkpoint_restore():
    cfg=replace(config(),sam2_trainable=True)
    branch=SAM2RGBEncoder(TinyImageEncoder(),64,2,trainable=True)
    branch.contract={'test':'same pretrained identity'}
    model=WanRGBDetector(None,branch,cfg).train()
    grids,latent,frame,_=inputs()
    result=model.forward_heads(grids,latent,torch.rand(2,5,3,32,32),frame)
    result['modification_map'].mean().backward()
    assert branch.image_encoder.projection.weight.grad.abs().sum()>0
    state=dict(heads=model.heads.state_dict(),rgb_encoder=branch.state_dict(),sam2_contract=branch.contract)
    second=WanRGBDetector(None,SAM2RGBEncoder(TinyImageEncoder(),64,2,True),cfg)
    second.rgb_encoder.contract=branch.contract
    restore_detector(second,state)
    torch.testing.assert_close(second.rgb_encoder.image_encoder.projection.weight,branch.image_encoder.projection.weight)
    second.rgb_encoder.contract={'test':'wrong weights'}
    with pytest.raises(ValueError,match='contract'):restore_detector(second,state)


def test_checkpoint_load_filters_encoder_and_requires_complete_weights(tmp_path):
    encoder=TinyImageEncoder()
    path=tmp_path/'sam.pt'
    state={'image_encoder.'+k:v.clone() for k,v in encoder.state_dict().items()}
    state['sam_mask_decoder.unused']=torch.ones(1)
    torch.save({'model':state},path)
    second=TinyImageEncoder();load_image_encoder_state(second,path)
    torch.testing.assert_close(second.projection.weight,encoder.projection.weight)
    state.pop('image_encoder.projection.bias');torch.save({'model':state},path)
    with pytest.raises(RuntimeError):load_image_encoder_state(second,path)


def test_cached_wan_path_still_runs_rgb_and_baseline_remains_optional():
    grids,latent,frame,_=inputs()
    branch=SAM2RGBEncoder(TinyImageEncoder(),64,2)
    model=WanRGBDetector(None,branch,config()).eval()
    batch=dict(video=torch.rand(2,5,3,32,32),frame_validity=frame)
    class Cache:
        def get_batch(self,batch):return grids,latent
    a=trial_forward(model,batch,Cache())
    b=model.forward_heads(grids,latent,batch['video'],frame)
    torch.testing.assert_close(a['modification_map'],b['modification_map'])
    base=DetectorHeads(replace(config(),rgb_branch='none')).eval()
    shared={k:v for k,v in model.heads.state_dict().items() if '.rgb_adapters.' not in k}
    base.load_state_dict(shared,strict=True)
    c=base(grids,latent,(5,32,32),frame)
    torch.testing.assert_close(a['visual_logit'],c['visual_logit'],rtol=0,atol=0)
    with pytest.raises(ValueError,match='RGB features'):model.heads(grids,latent,(5,32,32),frame)
    with pytest.raises(ValueError,match='RGB features'):base(grids,latent,(5,32,32),frame,inputs()[3])
