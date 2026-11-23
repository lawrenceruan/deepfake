"""Exercise the actual trainer/resume flow with tiny CPU extractors."""
from contextlib import nullcontext
import json
import sys
import torch
import pytest
from torch import nn
from deepfake import train
from deepfake.config import DetectorConfig
from deepfake.sam2_rgb import SAM2RGBEncoder
from deepfake.videomae_rgb import VideoMAERGBEncoder


class TinyWan(nn.Module):
    def forward(self,video,frames):
        grid=video.mean((1,2,3,4))[:,None,None,None,None].expand(-1,12,1,1,1)
        return [grid,grid,grid],frames[:,:1]


class TinySAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Conv2d(3,256,1)

    def forward(self,x):
        return {'backbone_fpn':[self.project(torch.nn.functional.avg_pool2d(x,s)) for s in (4,8,16)]}


class TinyVideoRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Conv2d(3,768,1)
        self.requires_grad_(False)
        self.contract={'weights':'video fixture'}
    def train(self,mode=True):
        return super().train(False)
    @torch.no_grad()
    def forward(self,video,frames):
        b,t,_,h,w=video.shape
        x=self.project(torch.nn.functional.avg_pool2d(video[:,::2].flatten(0,1),16))
        return [x.reshape(b,(t+1)//2,768,h//16,w//16)]


@pytest.mark.parametrize('branch',['sam2.1_hiera_large_decoder','videomae_base_decoder'])
def test_rgb_training_resumes_exactly_from_last_completed_epoch(tmp_path,monkeypatch,branch):
    torch.set_num_threads(2)
    # Only replace unavailable CUDA plumbing and heavyweight pretrained models.
    # Dataset loading, losses, optimizer, RNG restore and checkpoint IO are real.
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self)
    monkeypatch.setattr(nn.Module,'cuda',lambda self,*a,**k:self)
    monkeypatch.setattr(torch,'autocast',lambda *a,**k:nullcontext())
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'cpu-test')
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda state:None)
    monkeypatch.setattr(train.FrozenWanFeatures,'from_pretrained',lambda *a,**k:TinyWan())
    def sam(*args,**kwargs):
        branch=SAM2RGBEncoder(TinySAM(),64,1)
        branch.contract={'weights':'test fixture'}
        return branch
    monkeypatch.setattr(SAM2RGBEncoder,'from_pretrained',sam)
    monkeypatch.setattr(VideoMAERGBEncoder,'from_pretrained',lambda *a,**k:TinyVideoRGB())
    prefix='videomae' if branch=='videomae_base_decoder' else 'sam2'
    cfg=DetectorConfig(feature_layers=(0,1,2),backbone_dim=12,hidden_dim=16,
        attention_heads=4,decoder_channels=8,dropout=.2,rgb_branch=branch,sam2_image_size=64)
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg.to_dict()))
    rows=[]
    for split in ('train','val'):
        for label in (0,1):
            name=f'{split}_{label}';path=tmp_path/(name+'.pt')
            torch.save(dict(video=torch.full((1,3,32,32),.25+.5*label),map_target=torch.full((1,1,32,32),.1*label)),path)
            rows.append(dict(sample_id=name,source_video_id=name,tensor_path=str(path),split=split,
                visual_label=label,sample_type='paired_fake' if label else 'real',alignment_verified=True,
                map_convention='rgb_mean_half_dssim_gaussian11_sigma1.5'))
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def run(output,epochs,resume=None):
        argv=['deepfake.train','--config',str(config),'--checkpoint-dir',str(tmp_path/'wan'),
              '--context',str(tmp_path/'context.pt'),'--'+prefix+'-checkpoint',str(tmp_path/'rgb_weights'),
              '--manifest',str(manifest),'--output',str(output),'--epochs',str(epochs),
              '--accumulation','2','--cache-features']
        if resume:argv+=['--resume',str(resume)]
        monkeypatch.setattr(sys,'argv',argv);train.main()
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    run(full,2);run(resumed,1)
    original_run_config=(resumed/'run_config.json').read_bytes()
    run(resumed,2,resumed/'last.pt')
    a=torch.load(full/'last.pt',weights_only=True);b=torch.load(resumed/'last.pt',weights_only=True)
    assert a['epoch']==b['epoch']==1 and a[prefix+'_contract']==b[prefix+'_contract']
    assert 'rgb_encoder' not in b  # Frozen pretrained weights stay external.
    for name in a['heads']:torch.testing.assert_close(a['heads'][name],b['heads'][name],rtol=0,atol=0)
    assert a['epoch_metrics']==b['epoch_metrics']
    assert (resumed/'run_config.json').read_bytes()==original_run_config
    assert len((resumed/'metrics.jsonl').read_text().splitlines())==2
    assert all((resumed/name).exists() for name in ['last.pt','best.pt','best_accuracy.pt','best_auroc.pt'])
