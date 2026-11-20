import json
from pathlib import Path
import pytest
from deepfake.prepare_trial import split_candidates
from deepfake.data import audit_splits
from deepfake.train import classification_metrics


def test_component_split_prevents_transitive_source_donor_leaks():
    names=['001_002','002_003','003_001','004_005','005_004','006_007','008_009']
    rows=[dict(video_path=f'Deepfakes/{n}.mp4') for n in names]
    pools=split_candidates(rows,42)
    assert pools==split_candidates(rows,42)
    ids={k:{s for r in rs for s in Path(r['video_path']).stem.split('_')} for k,rs in pools.items()}
    assert ids['train'] and ids['val'] and not ids['train']&ids['val']
    assert sum(map(len,pools.values()))==len(rows)


def test_donor_to_target_leak_is_rejected(tmp_path):
    rows=[dict(source_video_id='a',donor_sequence_id='b',split='train'),
          dict(source_video_id='b',donor_sequence_id='c',split='val')]
    p=tmp_path/'manifest.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='leakage'):
        audit_splits(p)


def test_metrics_ties_and_perfect_ranking():
    tied=classification_metrics([0,1],[.5,.5])
    assert tied['auroc']==.5 and tied['accuracy']==.5
    perfect=classification_metrics([0,1,0,1],[.1,.8,.2,.9])
    assert perfect['auroc']==1 and perfect['accuracy']==1
