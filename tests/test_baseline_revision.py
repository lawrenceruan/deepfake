import json
import pytest
import torch
from deepfake.config import DetectorConfig
from deepfake.train import classification_metrics
from deepfake.checkpointing import atomic_save, commit_epoch, publish_epoch
from deepfake.data import audit_splits


def test_auc_retains_ranking_after_sigmoid_saturation():
    logits=[20.,40.];scores=torch.tensor(logits).sigmoid().tolist()
    assert scores == [1.,1.]
    assert classification_metrics([0,1],scores)['auroc']==.5
    fixed=classification_metrics([0,1],scores,logits)
    assert fixed['auroc']==1. and fixed['accuracy']==.5
    assert fixed['confident_errors_99']==1


def test_atomic_failure_preserves_existing_checkpoint(tmp_path,monkeypatch):
    import deepfake.checkpointing as c
    p=tmp_path/'last.pt';atomic_save({'epoch':2},p);before=p.read_bytes()
    def fail(value,handle):
        handle.write(b'partial');raise OSError('simulated interrupted write')
    monkeypatch.setattr(c.torch,'save',fail)
    with pytest.raises(OSError):atomic_save({'epoch':3},p)
    assert p.read_bytes()==before
    assert not list(tmp_path.glob('*.tmp'))


def test_epoch_commit_recovers_reports_and_best_checkpoint(tmp_path,monkeypatch):
    import deepfake.checkpointing as c
    state={'epoch':2,'heads':{'weight':torch.ones(2)},'epoch_metrics':{'epoch':2,'accuracy':.9},'epoch_predictions':[{'sample_id':'a'}],'checkpoint_updates':['best.pt','best_accuracy.pt']}
    actual=c.publish_epoch
    with monkeypatch.context() as m:
        m.setattr(c,'publish_epoch',lambda *args:(_ for _ in ()).throw(RuntimeError('preempted after commit')))
        with pytest.raises(RuntimeError):c.commit_epoch(tmp_path,state)
    loaded=torch.load(tmp_path/'last.pt',weights_only=True)
    actual(tmp_path,loaded);actual(tmp_path,loaded)
    assert len((tmp_path/'metrics.jsonl').read_text().splitlines())==1
    assert json.loads((tmp_path/'val_predictions_epoch_002.json').read_text())==state['epoch_predictions']
    assert torch.equal(torch.load(tmp_path/'best_accuracy.pt',weights_only=True)['heads']['weight'],torch.ones(2))


def test_recording_component_leakage_with_different_video_ids(tmp_path):
    rows=[{'sample_id':'a','source_video_id':'one','recording_component':'group','split':'train'}, {'sample_id':'b','source_video_id':'two','recording_component':'group','split':'val'}]
    p=tmp_path/'manifest.jsonl';p.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='leakage'):audit_splits(p)


@pytest.mark.parametrize('kwargs',[{'attention_heads':0},{'backbone_dim':0},{'mlp_ratio':0},{'num_train_timesteps':0}])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):DetectorConfig(**kwargs)


def test_config_normalizes_checkpoint_layer_lists():
    assert DetectorConfig(feature_layers=[9,19,29]).to_dict()==DetectorConfig().to_dict()
