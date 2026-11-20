import json
import torch
from torch import nn
import pytest
from deepfake import DetectorConfig, DetectorHeads, WanOnlyDetector, detector_loss
from deepfake.backbone import FrozenWanFeatures
from deepfake.losses import dssim_map
from deepfake.data import PreparedClipDataset, collate_clips, audit_splits

torch.set_num_threads(2)


def config():
    return DetectorConfig(feature_layers=(0,1,2), backbone_dim=12,
        hidden_dim=16, attention_heads=4, decoder_channels=8)


def features(batch=2, time=3):
    return [torch.randn(batch,12,time,2,2) for _ in range(3)]


def test_shapes_and_both_losses_reach_shared_modules():
    model = DetectorHeads(config())
    grids = features()
    valid = torch.ones(2,3,dtype=torch.bool)
    out = model(grids, valid, (9,32,32))
    assert out['visual_logit'].shape == (2,)
    assert out['modification_map'].shape == (2,9,1,32,32)
    assert torch.isfinite(out['modification_map']).all()
    target = torch.rand_like(out['modification_map'])
    loss = detector_loss(out, torch.tensor([0.,1.]), target,
        torch.ones_like(target,dtype=torch.bool), torch.ones(2,9,dtype=torch.bool))
    for name in ['classification','map']:
        model.zero_grad()
        loss[name].backward(retain_graph=True)
        assert model.fusion.fuse.weight.grad.abs().sum() > 0
        assert model.blocks[0].temporal.in_proj_weight.grad.abs().sum() > 0


def test_padding_cannot_change_valid_outputs():
    model = DetectorHeads(config()).eval()
    grids = features()
    valid = torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)
    frames = torch.tensor([[1]*9,[1]*5+[0]*4],dtype=torch.bool)
    a = model(grids,valid,(9,32,32),frames)
    changed = [g.clone() for g in grids]
    for g in changed:
        g[1,:,2] = 1e5
    b = model(changed,valid,(9,32,32),frames)
    torch.testing.assert_close(a['visual_logit'],b['visual_logit'])
    torch.testing.assert_close(a['modification_map'],b['modification_map'])
    assert not b['modification_map'][1,5:].any()
    single = model([g[1:2,:,:2] for g in grids],valid[1:2,:2],(5,32,32))
    torch.testing.assert_close(a['modification_map'][1:2,:5],single['modification_map'])


def test_masked_loss_and_no_supervision():
    pred = torch.rand(1,3,1,8,8,requires_grad=True)
    mask = torch.zeros_like(pred,dtype=torch.bool)
    mask[:,:1] = True
    target = torch.full_like(pred,float('nan'))
    target[:,:1] = 0
    out = {'visual_logit':torch.zeros(1,requires_grad=True),'modification_map':pred}
    a = detector_loss(out,torch.ones(1),target,mask,torch.ones(1,3,dtype=torch.bool))
    a['loss'].backward()
    assert torch.isfinite(a['loss']) and not pred.grad[:,1:].any()
    mask.zero_()
    assert detector_loss(out,torch.ones(1),target,mask,torch.ones(1,3,dtype=torch.bool))['map'] == 0


def test_dssim_identity_and_local_edit():
    real = torch.rand(2,3,32,32)
    assert dssim_map(real,real).abs().max() < 1e-5
    fake = real.clone()
    fake[:,:,12:20,12:20] = 0
    result = dssim_map(real,fake)
    assert result[:,:,12:20,12:20].mean() > 0
    assert result[:,:,:5,:5].abs().max() < 1e-5


class FakeVAE:
    def __init__(self):
        self.model = nn.Conv3d(3,4,1)
    def encode(self,videos):
        return [self.model(v[None])[:, :, ::4, ::16, ::16][0] for v in videos]


class FakeWan(nn.Module):
    text_len, text_dim, dim, patch_size = 8, 6, 12, (1,2,2)
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv3d(4,12,(1,2,2),stride=(1,2,2))
        self.blocks = nn.ModuleList([nn.Linear(12,12) for _ in range(3)])
    def forward(self,x,t,context,seq_len,feature_layers):
        grid = self.patch(x[0][None])
        shape = grid.shape[2:]
        tokens = grid.flatten(2).transpose(1,2)
        # Global mixing exposes accidental contamination by padded inputs.
        tokens = tokens + tokens.mean(1,keepdim=True)
        outputs = []
        for i,block in enumerate(self.blocks):
            tokens = block(tokens)
            if i in feature_layers:
                outputs.append(tokens)
        return dict(hidden_states=outputs,grid_sizes=torch.tensor([shape]),seq_lens=torch.tensor([tokens.shape[1]]))


def test_frozen_extractor_and_padding_independence():
    extractor = FrozenWanFeatures(FakeVAE(),FakeWan(),torch.randn(2,6),config())
    model = WanOnlyDetector(extractor,config()).train()
    assert not extractor.training and not extractor.backbone.training
    assert not any(p.requires_grad for p in extractor.parameters())
    video = torch.rand(2,9,3,32,32)
    valid = torch.tensor([[1]*9,[1]*5+[0]*4],dtype=torch.bool)
    a = model(video,valid)
    video[1,5:] = torch.rand_like(video[1,5:])
    b = model(video,valid)
    torch.testing.assert_close(a['visual_logit'],b['visual_logit'])
    b['visual_logit'].sum().backward()
    assert all(p.grad is None for p in extractor.parameters())
    assert model.heads.classifier[-1].weight.grad is not None


def test_cached_frozen_features_match_live_and_allow_head_gradients():
    from deepfake.train import trial_forward
    extractor = FrozenWanFeatures(FakeVAE(),FakeWan(),torch.randn(2,6),config())
    model = WanOnlyDetector(extractor,config()).train()
    batch = dict(video=torch.rand(1,5,3,32,32),frame_validity=torch.ones(1,5,dtype=torch.bool),
                 metadata=[dict(sample_id='clip1',split='train')])
    direct = trial_forward(model,batch)
    cache = {}
    first = trial_forward(model,batch,cache)
    # A cache hit must bypass the extractor completely.
    def forbidden(*args,**kwargs):
        raise AssertionError('Extractor called on cache hit')
    extractor.forward = forbidden
    second = trial_forward(model,batch,cache)
    torch.testing.assert_close(direct['visual_logit'],second['visual_logit'])
    torch.testing.assert_close(first['modification_map'],second['modification_map'])
    second['visual_logit'].sum().backward()
    assert model.heads.fusion.fuse.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in extractor.parameters())


def test_prepared_data_and_split_leakage(tmp_path):
    torch.save({'video':torch.rand(5,3,32,32)},tmp_path/'real.pt')
    torch.save({'video':torch.rand(9,3,32,32)},tmp_path/'fake.pt')
    rows = [dict(sample_id='real',tensor_path='real.pt',split='train',visual_label=0,
                 sample_type='real',source_video_id='a'),
            dict(sample_id='fake',tensor_path='fake.pt',split='train',visual_label=1,
                 sample_type='unpaired_fake',source_video_id='b')]
    path = tmp_path/'manifest.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    ds = PreparedClipDataset(path,'train')
    batch = collate_clips([ds[0],ds[1]])
    assert batch['video'].shape == (2,9,3,32,32)
    assert not batch['frame_validity'][0,5:].any()
    assert not batch['map_validity'][1].any()
    assert not batch['map_target'][0].any()
    audit_splits(path)
    rows.append(dict(rows[0],sample_id='leak',split='val'))
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='leakage'):
        audit_splits(path)


def test_heads_small_subset_overfit_and_serialization(tmp_path):
    torch.manual_seed(1)
    model = DetectorHeads(config())
    grids = features(batch=1,time=2)
    valid = torch.ones(1,2,dtype=torch.bool)
    frames = torch.ones(1,5,dtype=torch.bool)
    target = torch.zeros(1,5,1,32,32)
    opt = torch.optim.Adam(model.parameters(),lr=0.01)
    losses = []
    for _ in range(25):
        out = model(grids,valid,(5,32,32))
        loss = detector_loss(out,torch.zeros(1),target,torch.ones_like(target,dtype=torch.bool),frames)['loss']
        opt.zero_grad();loss.backward();opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]*0.35
    path = tmp_path/'heads.pt'
    torch.save(model.state_dict(),path)
    restored = DetectorHeads(config())
    restored.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(model(grids,valid,(5,32,32))['visual_logit'],
                               restored(grids,valid,(5,32,32))['visual_logit'])
