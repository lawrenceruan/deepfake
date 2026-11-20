import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from deepfake import preprocess as prep
from deepfake.data import PreparedClipDataset
from deepfake.losses import dssim_map


def test_center_crop_preserves_pixels_and_rejects_unsupported_geometry():
    rng = np.random.default_rng(3)
    real = rng.integers(0,256,(80,120,3),dtype=np.uint8)
    fake = real[:,8:-8].copy()
    box = prep.center_geometry(real.shape, fake.shape)
    assert box == [8,0,112,80]
    assert np.array_equal(prep.crop(real,box),fake)
    assert not np.array_equal(cv2.resize(real,(104,80)),fake)
    for shape in [(81,104,3),(80,121,3)]:
        with pytest.raises(ValueError,match='Unsupported geometry'):
            prep.center_geometry(real.shape,shape)


def test_alignment_rejects_shifted_time_and_accepts_local_face_edit():
    rng = np.random.default_rng(5)
    real = {i:rng.integers(0,256,(96,128,3),dtype=np.uint8) for i in range(5)}
    fake = real[2].copy()
    fake[35:60,45:75] = 0
    assert prep.check_correspondence(real,{2:fake},[2],[40,30,80,65])['passed']
    assert not prep.check_correspondence(real,{2:real[3]},[2],[40,30,80,65])['passed']
    flat = {i:np.zeros_like(fake) for i in range(5)}
    assert not prep.check_correspondence(flat,{2:flat[2]},[2],[40,30,80,65])['passed']


def test_prepare_pair_roundtrip_raw_unchanged_and_masked_borders(tmp_path,monkeypatch):
    root, out = tmp_path/'raw', tmp_path/'prepared'
    for folder in [root/'original',root/'Face2Face',out/'clips',out/'reports',out/'previews']:
        folder.mkdir(parents=True)
    real_path, fake_path = root/'original/001.mp4', root/'Face2Face/001_002.mp4'
    real_path.write_bytes(b'raw real sentinel')
    fake_path.write_bytes(b'raw fake sentinel')
    rng=np.random.default_rng(7)
    real={i:rng.integers(0,256,(96,144,3),dtype=np.uint8) for i in range(7)}
    fake={i:im[:,8:-8].copy() for i,im in real.items()}
    for im in fake.values():
        im[40:50,50:60]=255
    def decode(path,indices,required_indices=None):
        data=real if path==real_path else fake
        return {i:data[i] for i in indices}, {i:i/25 for i in indices}, dict(fps=25,header_frames=7)
    monkeypatch.setattr(prep,'decode_indices',decode)
    monkeypatch.setattr(prep,'face_tube',lambda frames,margin:([32,16,96,80],[]))
    args=SimpleNamespace(start_frame=2,frame_stride=1,frames=3,face_margin=1.4,
                         max_background_mae=.04,max_background_p95=.12,size=64,split='smoke')
    rows,report=prep.prepare_pair(root,out,dict(reference_path='original/001.mp4',video_path='Face2Face/001_002.mp4'),args,0)
    assert report['status']=='accepted' and report['raw_inputs_unchanged']
    assert real_path.read_bytes()==b'raw real sentinel'
    assert fake_path.read_bytes()==b'raw fake sentinel'
    manifest=out/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    ds=PreparedClipDataset(manifest,'smoke')
    f,r=ds[0],ds[1]
    assert f['video'].shape==(3,3,64,64)
    assert torch.allclose(f['map_target'],dssim_map(r['video'],f['video']))
    assert f['map_target'].max()>0
    assert not r['map_target'].any()
    assert not f['map_validity'][...,:5,:].any()
    assert f['map_validity'][...,5:-5,5:-5].all()


def test_input_paths_cannot_escape_raw_root(tmp_path):
    root=tmp_path/'raw'
    root.mkdir()
    (tmp_path/'outside.mp4').touch()
    with pytest.raises(ValueError,match='outside raw root'):
        prep.resolve_input(root,'../outside.mp4')


def test_output_cannot_overlap_raw_data_or_overwrite_existing_run(tmp_path):
    root=tmp_path/'raw'
    root.mkdir()
    for out in (root,root/'generated',tmp_path):
        with pytest.raises(ValueError,match='disjoint'):
            prep.validate_output(root,out)
    existing=tmp_path/'existing'
    existing.mkdir()
    with pytest.raises(ValueError,match='already exists'):
        prep.validate_output(root,existing)
    link=tmp_path/'alias'
    link.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError,match='disjoint'):
        prep.validate_output(root,link/'generated')
    prep.validate_output(root,tmp_path/'new_run')
