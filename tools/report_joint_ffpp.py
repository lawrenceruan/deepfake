"""Report joint-run metrics separately for each manipulation method."""
import argparse
import json
from pathlib import Path
from deepfake.train import classification_metrics


def summarize(predictions, metadata):
    def weighted_map(rows):
        # All prepared frames share the same spatial validity mask. Weighting
        # clip MAE by valid frame count is therefore valid-pixel weighting.
        n=sum(metadata[r['sample_id']]['segment']['valid_frames'] for r in rows)
        return sum(r['map_mae']*metadata[r['sample_id']]['segment']['valid_frames']
                   for r in rows)/n if n else None
    fake=[r for r in predictions if r['label']==1]
    return dict(clips=len(predictions),fake_clips=len(fake),real_clips=len(predictions)-len(fake),
        recording_groups=len({metadata[r['sample_id']]['recording_component'] for r in predictions}),
        map_mae=weighted_map(predictions),fake_map_mae=weighted_map(fake),
        **classification_metrics([r['label'] for r in predictions],
                                 [r['fake_probability'] for r in predictions],
                                 logits=[r['visual_logit'] for r in predictions] if all('visual_logit' in r for r in predictions) else None))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.run/'run_config.json').read_text())
    metadata={r['sample_id']:r for line in Path(cfg['manifest']).read_text().splitlines()
              for r in [json.loads(line)] if r['split']=='val'}
    families=sorted({r['generator_family'] for r in metadata.values()})
    metrics=[json.loads(line) for line in (a.run/'metrics.jsonl').read_text().splitlines()]
    by_epoch=[]
    for metric in metrics:
        epoch=metric['epoch']
        predictions=json.loads((a.run/f'val_predictions_epoch_{epoch:03d}.json').read_text())
        assert {r['sample_id'] for r in predictions}==set(metadata)
        by_epoch.append(dict(epoch=epoch,methods={family:summarize(
            [r for r in predictions if metadata[r['sample_id']]['generator_family']==family],metadata)
            for family in families}))
    (a.run/'per_method_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in by_epoch))
    verified=json.loads((a.run/'verified_evaluation.json').read_text())
    selected=next(r for r in by_epoch if r['epoch']+1==verified['checkpoint_epoch'])
    result=dict(checkpoint_epoch=verified['checkpoint_epoch'],combined=verified,
        methods=selected['methods'],heldout_recording_groups=50,
        evaluation_note='Validation entries share 50 recording groups across methods. This validation set selects the checkpoint.',
        sampling_policy='Uniform shuffled sampling over all manifest entries, preserving each method\'s paired real entries; no class or method reweighting.')
    (a.run/'joint_evaluation_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
