"""Exercise the real Wan forward method with tiny random weights on CPU.

Only the CUDA FlashAttention kernel is replaced by PyTorch SDPA. This is not
a pretrained 5B or CUDA integration test.
"""
import torch
import pytest
from torch.nn import functional as F
from wan.modules import model as wan_model


def cpu_attention(q,k,v,k_lens=None,**kwargs):
    mask = None
    if k_lens is not None:
        mask = torch.arange(k.shape[1])[None,None,None,:] < k_lens[:,None,None,None]
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),
        v.transpose(1,2),attn_mask=mask).transpose(1,2).contiguous()


def test_features_match_block_hooks_and_generation_stays_unchanged(monkeypatch):
    monkeypatch.setattr(wan_model,'flash_attention',cpu_attention)
    model = wan_model.WanModel(model_type='ti2v',in_dim=4,out_dim=4,dim=24,
        ffn_dim=48,freq_dim=8,text_dim=6,text_len=4,num_heads=2,num_layers=3).eval()
    videos = [torch.randn(4,3,4,4),torch.randn(4,2,4,4)]
    args = dict(x=videos,t=torch.zeros(2),context=[torch.randn(2,6),torch.randn(2,6)],seq_len=12)
    seen = []
    handles = [b.register_forward_hook(lambda m,a,o:seen.append(o.detach().clone())) for b in model.blocks]
    normal = model(**args)
    for handle in handles:
        handle.remove()
    features = model(**args,feature_layers=(2,0))
    torch.testing.assert_close(features['hidden_states'][0],seen[2])
    torch.testing.assert_close(features['hidden_states'][1],seen[0])
    assert features['grid_sizes'].tolist() == [[3,2,2],[2,2,2]]
    assert features['seq_lens'].tolist() == [12,8]
    again = model(**args)
    for a,b in zip(normal,again):
        torch.testing.assert_close(a,b)
    assert [tuple(v.shape) for v in normal] == [(4,3,4,4),(4,2,4,4)]
    # The feature-only path must not execute the generator output head.
    def fail(*args):
        raise AssertionError('Generator head executed')
    monkeypatch.setattr(model.head,'forward',fail)
    model(**args,feature_layers=(0,))
    with pytest.raises(ValueError):
        model(**args,feature_layers=(3,))
