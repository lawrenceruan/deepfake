import json
import pytest
import torch
from deepfake.prepare_full_segments import compact_payload, split_rows, split_rows_from_reference
from deepfake.data import PreparedClipDataset, audit_splits


def test_compact_roundtrip_and_loader(tmp_path):
    video=torch.randint(0,256,(5,3,32,32),dtype=torch.uint8).float()/255
    payload=dict(video=video,map_target=torch.zeros(5,1,32,32),
                 map_validity=torch.ones(5,1,32,32,dtype=torch.bool),
                 frame_validity=torch.tensor([True,True,True,False,False]))
    compact=compact_payload(payload,real=True)
    assert compact['video'].dtype==torch.uint8 and 'map_target' not in compact
    torch.save(compact,tmp_path/'sample.pt')
    (tmp_path/'manifest.jsonl').write_text(json.dumps(dict(sample_id='x',tensor_path='sample.pt',
        split='train',sample_type='real',visual_label=0)))
    sample=PreparedClipDataset(tmp_path/'manifest.jsonl','train')[0]
    assert torch.equal(sample['video'],video)
    assert sample['map_target'].count_nonzero()==0
    assert not sample['map_validity'][3:].any()
    with pytest.raises(ValueError,match='change RGB'):
        compact_payload(dict(payload,video=torch.full_like(video,.12345)))


def test_component_split_reserves_every_relative(tmp_path):
    pairs=[];rows=[]
    for group in range(60):
        a,b,c=[str(group*3+i) for i in range(3)]
        for source,donor in [(a,b),(b,c)]:
            pairs.append(dict(video_path=f'Deepfakes/{source}_{donor}.mp4'))
            for segment in range(3):
                for label in (0,1):
                    rows.append(dict(sample_id=f'{source}_{donor}_{segment}_{label}',source_video_id=source,
                        donor_sequence_id=donor,visual_label=label,
                        raw_video_path=f'Deepfakes/{source}_{donor}.mp4' if label else f'original/{source}.mp4',
                        segment=dict(valid_frames=25 if segment<2 else 5,maximum_frames=25)))
    active,reserved,split=split_rows(rows,pairs,50,42)
    val=[r for r in active if r['split']=='val']
    train=[r for r in active if r['split']=='train']
    assert len(val)==50 and sum(r['visual_label'] for r in val)==25
    assert len({r['recording_component'] for r in val})==50
    assert len({r['raw_video_path'] for r in val})==50
    assert all(r['segment']['valid_frames']==25 for r in val)
    assert len(active)+len(reserved)==len(rows)
    assert not {r['recording_component'] for r in train}&{r['recording_component'] for r in val+reserved}
    assert split_rows(rows,pairs,50,42)==(active,reserved,split)
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in active))
    audit_splits(manifest)


def test_cross_method_holdout_never_replaces_missing_group(tmp_path):
    pairs=[];rows=[]
    for group in range(8):
        source,donor=str(group*2),str(group*2+1)
        pairs.append(dict(video_path=f'Deepfakes/{source}_{donor}.mp4'))
        for segment in range(2):
            for label in (0,1):
                rows.append(dict(sample_id=f'Deepfakes_{source}_{segment}_{label}',source_video_id=source,
                    donor_sequence_id=donor,visual_label=label,
                    raw_video_path=f'Deepfakes/{source}_{donor}.mp4' if label else f'original/{source}.mp4',
                    segment=dict(segment_index=segment,valid_frames=25,maximum_frames=25)))
    reference,_,base=split_rows(rows,pairs,4,42)
    missing=base['selected_components'][0]
    other=[dict(r,sample_id=r['sample_id'].replace('Deepfakes','FaceSwap'),
                raw_video_path=r['raw_video_path'].replace('Deepfakes','FaceSwap'))
           for r in rows if r['source_video_id']!=missing]
    active,reserved,split=split_rows_from_reference(other,pairs,reference,4,42)
    assert split['selected_components']==base['selected_components']
    assert split['evaluation_clips']==3 and split['missing_evaluation_components'][0]['component']==missing
    assert all(r['same_parent_and_segment'] for r in split['matched_reference_clips'])
    assert not {r['recording_component'] for r in active if r['split']=='train'}&set(base['selected_components'])
    assert len(active)+len(reserved)==len(other)
    manifest=tmp_path/'combined.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in reference+active))
    audit_splits(manifest)
    bad=[dict(r,recording_component='wrong') if r['split']=='val' else r for r in reference]
    with pytest.raises(ValueError,match='components differ'):
        split_rows_from_reference(other,pairs,bad,4,42)
