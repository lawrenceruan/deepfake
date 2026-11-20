import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepfake import preprocess as prep
from deepfake.data import PreparedClipDataset
from deepfake.losses import dssim_map
from deepfake.segment_preprocess import plan_segments, prepare_segment, read_timestamps, normalize_pairs
from deepfake.preprocessing_config import PreprocessingConfig


def test_full_segments_and_short_tail_preserve_sampling_rate():
    times=np.arange(200)/25  # 7.96 seconds of timestamp coverage.
    plans=plan_segments(times)
    assert [p['segment_start_s'] for p in plans]==[0.,3.,6.]
    assert [p['valid_frames'] for p in plans]==[25,25,16]
    for p in plans:
        assert p['maximum_frames']==25
        np.testing.assert_allclose(np.diff(p['requested_timestamps_s']),.125)
        assert max(abs(np.array(p['selected_reference_timestamps_s'])-p['requested_timestamps_s']))<=.02+1e-8
    assert plans[0]['frame_indices'][-1]==plans[1]['frame_indices'][0]


def test_short_and_low_fps_sources_are_not_stretched():
    p=plan_segments(10+np.arange(50)/25)[0]
    assert p['valid_frames']==16
    assert p['requested_timestamps_s'][-1]==11.875
    assert len(plan_segments(np.arange(76)/25))==1  # No duplicate endpoint-only tail.
    low=plan_segments(np.arange(7)/2)[0]
    assert low['repeated_source_samples']>0
    assert plan_segments([0.])[0]['valid_frames']==1
    with pytest.raises(ValueError):
        plan_segments([0,0,1])


def test_short_segment_roundtrip_and_dssim_excludes_padding(tmp_path,monkeypatch):
    root,out=tmp_path/'raw',tmp_path/'prepared'
    for p in (root/'original',root/'Face2Face',out/'clips',out/'reports',out/'previews'):
        p.mkdir(parents=True)
    rp,fp=root/'original/001.mp4',root/'Face2Face/001_002.mp4'
    rp.write_bytes(b'real');fp.write_bytes(b'fake')
    rng=np.random.default_rng(3)
    real={i:rng.integers(0,256,(96,144,3),dtype=np.uint8) for i in range(25)}
    fake={i:im[:,8:-8].copy() for i,im in real.items()}
    for im in fake.values():
        im[40:50,50:60]=255
    def decode(path,indices,required_indices=None):
        data=real if path==rp else fake
        return {i:data[i] for i in indices if i in data},{i:i/25 for i in data},dict(fps=25,header_frames=25)
    monkeypatch.setattr(prep,'decode_indices',decode)
    monkeypatch.setattr(prep,'face_tube',lambda frames,margin:([32,16,96,80],[]))
    args=SimpleNamespace(face_margin=1.4,max_background_mae=.04,max_background_p95=.12,size=64,split='smoke')
    plan=plan_segments(np.arange(25)/25)[0]
    samples,report=prepare_segment(root,out,dict(reference_path='original/001.mp4',video_path='Face2Face/001_002.mp4'),args,0,plan)
    assert report['status']=='accepted'
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
    ds=PreparedClipDataset(out/'manifest.jsonl','smoke')
    f,r=ds[0],ds[1]
    n=plan['valid_frames']
    assert n==8 and f['video'].shape==(25,3,64,64)
    assert f['frame_validity'].sum()==n
    assert not f['map_validity'][n:].any()
    assert not f['map_target'][n:].any()
    assert torch.equal(f['video'][n:],f['video'][n-1:n].expand(25-n,-1,-1,-1))
    torch.testing.assert_close(f['map_target'][:n],dssim_map(r['video'][:n],f['video'][:n]))
    assert rp.read_bytes()==b'real' and fp.read_bytes()==b'fake'


def test_invalid_decoded_timestamps_rejected(monkeypatch):
    class Capture:
        def __init__(self,*a):self.n=0
        def isOpened(self):return True
        def read(self):
            self.n+=1
            return (self.n<=2,None)
        def get(self,key):return 0.
        def release(self):pass
    monkeypatch.setattr('deepfake.segment_preprocess.cv2.VideoCapture',Capture)
    with pytest.raises(ValueError,match='timestamps'):
        read_timestamps('unused.mp4')


def test_configuration_changes_sampling_and_is_fingerprinted(tmp_path):
    three=PreprocessingConfig()
    five=PreprocessingConfig(segment_duration_s=5.)
    assert three.maximum_frames==25 and five.maximum_frames==41
    assert three.fingerprint!=five.fingerprint
    p=tmp_path/'config.json'
    p.write_text(json.dumps(five.to_dict()))
    loaded=PreprocessingConfig.load(p)
    assert loaded==five
    plans=plan_segments(np.arange(251)/25,loaded.segment_duration_s,loaded.sampling_fps)
    assert len(plans)==2 and all(p['maximum_frames']==41 for p in plans)
    with pytest.raises(ValueError):
        PreprocessingConfig(segment_duration_s=float('nan'))


def test_reprocessing_manifest_preserves_recording_splits():
    fake=dict(tensor_path='old.pt',sample_type='paired_fake',raw_video_path='Deepfakes/001_002.mp4',
              raw_reference_path='original/001.mp4',split='val')
    rows=normalize_pairs([fake,dict(fake,tensor_path='old_other_clip.pt'),dict(fake,sample_type='real')])
    assert len(rows)==1 and rows[0]['split']=='val'
    assert rows[0]['video_path']=='Deepfakes/001_002.mp4'
    with pytest.raises(ValueError,match='conflicting splits'):
        normalize_pairs([fake,dict(fake,split='train')])


def test_loader_rejects_mixed_or_outdated_preprocessing(tmp_path):
    rows=[dict(sample_id='a',split='train',preprocessing_config_id='old')]
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='regenerate'):
        PreparedClipDataset(manifest,'train',expected_preprocessing_id=PreprocessingConfig().fingerprint)
    manifest.write_text('\n'.join(json.dumps(r) for r in rows+[dict(rows[0],sample_id='b',preprocessing_config_id='new')]))
    with pytest.raises(ValueError,match='Mixed preprocessing'):
        PreparedClipDataset(manifest,'train')
