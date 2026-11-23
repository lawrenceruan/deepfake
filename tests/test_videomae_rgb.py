from types import SimpleNamespace
from dataclasses import replace
import pytest
import torch
from torch import nn
from deepfake.config import DetectorConfig
from deepfake.model import DetectorHeads, WanRGBDetector, build_detector, restore_detector
from deepfake.modules import VideoMAEMapDecoder
from deepfake.videomae_rgb import VideoMAERGBEncoder

torch.set_num_threads(2)


def tiny_encoder():
    transformers=pytest.importorskip('transformers')
    cfg=transformers.VideoMAEConfig(image_size=32,num_frames=4,patch_size=16,tubelet_size=2,
        hidden_size=768,num_hidden_layers=1,num_attention_heads=12,intermediate_size=32,
        use_mean_pooling=False)
    return transformers.VideoMAEModel(cfg).eval()


def cfg():
    return DetectorConfig(feature_layers=(0,1,2),backbone_dim=12,hidden_dim=16,
        attention_heads=4,decoder_channels=8,rgb_branch='videomae_base_decoder',videomae_image_size=32)


def test_native_shape_matches_official_forward_and_frozen_padding():
    encoder=tiny_encoder()
    branch=VideoMAERGBEncoder(encoder,32).train()
    assert not branch.training and not encoder.training
    video=torch.rand(1,4,3,32,32)
    with torch.no_grad():
        expected=encoder(pixel_values=(video-branch.mean)/branch.std).last_hidden_state
    actual=branch(video,torch.ones(1,4,dtype=torch.bool))[0]
    tokens=actual.permute(0,1,3,4,2).reshape(1,-1,768)
    torch.testing.assert_close(tokens,expected)
    videos=torch.rand(2,5,3,32,32)
    videos[1,1:]=float('nan')
    valid=torch.tensor([[1,1,1,1,1],[1,0,0,0,0]],dtype=torch.bool)
    result=branch(videos,valid)[0]
    assert result.shape==(2,3,768,2,2)
    assert not result.requires_grad and not result[1,1:].any()
    single=branch(videos[1:2,:1],valid[1:2,:1])[0]
    torch.testing.assert_close(single,result[1:2,:1])
    assert all(not p.requires_grad for p in branch.parameters())


def test_spatial_position_adaptation_and_invalid_inputs():
    branch=VideoMAERGBEncoder(tiny_encoder(),48)
    out=branch(torch.rand(1,3,3,32,32),torch.ones(1,3,dtype=torch.bool))[0]
    assert out.shape==(1,2,768,3,3) and torch.isfinite(out).all()
    with pytest.raises(ValueError,match='right-padded'):
        branch(torch.rand(1,3,3,32,32),torch.tensor([[1,0,1]],dtype=torch.bool))
    with pytest.raises(ValueError,match='finite RGB'):
        branch(torch.full((1,1,3,32,32),2.),torch.ones(1,1,dtype=torch.bool))


def test_tubelet_interpolation_preserves_centers_and_odd_last_frame():
    x=torch.tensor([.5,2.5,4.]).reshape(1,1,3,1,1)
    out=VideoMAEMapDecoder.restore_rgb_time(x,5).flatten()
    torch.testing.assert_close(out,torch.tensor([.5,1.,2.,3.,4.]))
    torch.testing.assert_close(VideoMAEMapDecoder.restore_rgb_time(x[:,:,:1],1),x[:,:,:1])
    even=VideoMAEMapDecoder.restore_rgb_time(x[:,:,:2],4).flatten()
    torch.testing.assert_close(even,torch.tensor([.5,1.,2.,2.5]))


def test_compact_rgb_affects_maps_only_and_gradients_reach_all_adapters():
    torch.manual_seed(3)
    heads=DetectorHeads(cfg()).eval()
    grids=[torch.randn(2,12,2,2,2) for _ in range(3)]
    latent=torch.tensor([[1,1],[1,0]],dtype=torch.bool)
    frames=torch.tensor([[1,1,1,1,1],[1,0,0,0,0]],dtype=torch.bool)
    rgb=[torch.randn(2,3,768,2,2)]
    rgb[0][1,1:]=float('nan')
    a=heads(grids,latent,(5,32,32),frames,rgb)
    b=heads(grids,latent,(5,32,32),frames,[rgb[0]+torch.randn_like(rgb[0])])
    torch.testing.assert_close(a['visual_logit'],b['visual_logit'],rtol=0,atol=0)
    assert not torch.allclose(a['modification_map'][0],b['modification_map'][0])
    assert torch.isfinite(a['modification_map']).all() and not a['modification_map'][1,1:].any()
    a['modification_map'].mean().backward()
    assert all(x[0].weight.grad.abs().sum()>0 for x in heads.decoder.rgb_adapters)
    assert heads.fusion.fuse.weight.grad.abs().sum()>0
    assert heads.classifier[-1].weight.grad is None
    with pytest.raises(ValueError,match='RGB checkpoint'):
        build_detector(None,replace(cfg(),rgb_branch='none'),videomae_checkpoint='unused')


def test_restore_requires_identical_videomae_contract():
    encoder=nn.Identity();encoder.contract={'weights':'first'}
    model=WanRGBDetector(None,encoder,cfg())
    state={'heads':model.heads.state_dict(),'videomae_contract':{'weights':'first'}}
    restore_detector(model,state)
    state['videomae_contract']={'weights':'changed'}
    with pytest.raises(ValueError,match='contract'):restore_detector(model,state)
