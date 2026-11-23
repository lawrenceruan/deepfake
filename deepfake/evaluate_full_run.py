"""Reload a selected checkpoint, verify all holdout predictions, export examples."""
import argparse
import json
from pathlib import Path
import torch
from .config import DetectorConfig
from .backbone import FrozenWanFeatures
from .model import build_detector, restore_detector
from .data import PreparedClipDataset, collate_clips, audit_splits
from .train import classification_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint', choices=['best.pt','last.pt','best_accuracy.pt','best_auroc.pt'], default='best.pt')
    p.add_argument('--output', type=Path, help='Separate report directory; required for non-default checkpoints')
    p.add_argument('--videomae-checkpoint', help='Override the VideoMAE pretrained directory')
    p.add_argument('--sam2-checkpoint', help='Override the RGB checkpoint location')
    a=p.parse_args()
    if a.checkpoint != 'best.pt' and a.output is None:
        p.error('--output is required for a non-default checkpoint')
    report = a.output or a.run
    if a.output is not None:
        report.mkdir(parents=True, exist_ok=False)
    state=torch.load(a.run/a.checkpoint,weights_only=True,map_location='cpu')
    cfg=DetectorConfig(**state['config'])
    audit_splits(state['manifest'])
    ds=PreparedClipDataset(state['manifest'],'val')
    recorded={r['sample_id']:r for r in json.loads((a.run/f"val_predictions_epoch_{state['epoch']:03d}.json").read_text())}
    selected=set()
    reasons={}
    for label,positions in [(1,[0,.5,1]),(0,[.5,1])]:
        rows=sorted((r for r in recorded.values() if r['label']==label),key=lambda r:(r['map_mae'],r['sample_id']))
        for position in positions:
            row=rows[round(position*(len(rows)-1))]
            selected.add(row['sample_id'])
            reasons[row['sample_id']]=f"{'fake' if label else 'real'} map-MAE quantile {position}"
    errors=sorted((r for r in recorded.values() if (r['fake_probability']>=.5)!=bool(r['label'])),
                  key=lambda r:abs(r['fake_probability']-r['label']),reverse=True)[:2]
    for row in errors:
        selected.add(row['sample_id']);reasons[row['sample_id']]='most confident classification error'
    model=build_detector(FrozenWanFeatures.from_pretrained(state['checkpoint_dir'],state['context_path'],cfg),cfg,
                         a.sam2_checkpoint or state.get('sam2_checkpoint'),
                         videomae_checkpoint=a.videomae_checkpoint or state.get('videomae_checkpoint'))
    restore_detector(model,state);model.heads.cuda();model.eval()
    all_rows=[json.loads(s) for s in (Path(state['manifest']).parent/'all_prepared_manifest.jsonl').read_text().splitlines()]
    references={r['pair_id']:r for r in all_rows if r['visual_label']==0}
    results=[];visuals=[];logits=[]
    total_error=0.;total_pixels=0;fake_error=0.;fake_pixels=0;zero_error=0.
    with torch.no_grad():
        for i in range(len(ds)):
            item=ds[i]; row=item['metadata']
            batch=collate_clips([item])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(batch['video'].cuda(),batch['frame_validity'].cuda())
            logit=float(pred['visual_logit'].float()[0]);logits.append(logit)
            probability=float(pred['visual_logit'].float().sigmoid()[0])
            maps=pred['modification_map'][0].float().cpu()
            mask=item['map_validity'];target=item['map_target']
            error=float((maps-target).abs()[mask].sum());pixels=int(mask.sum())
            total_error+=error;total_pixels+=pixels
            if int(row['visual_label']):
                fake_error+=error;fake_pixels+=pixels;zero_error+=float(target[mask].sum())
            previous=recorded[row['sample_id']]
            assert abs(probability-previous['fake_probability'])<1e-5,(row['sample_id'],probability,previous)
            assert abs(error/pixels-previous['map_mae'])<1e-5
            results.append(dict(sample_id=row['sample_id'],label=row['visual_label'],
                                fake_probability=probability,visual_logit=logit,map_mae=error/pixels))
            if row['sample_id'] in selected:
                n=int(item['frame_validity'].sum());indices=sorted(set([0,n//2,n-1]))
                reference_row=references[row['pair_id']]
                raw=torch.load(Path(state['manifest']).parent/reference_row['tensor_path'],weights_only=True,map_location='cpu')
                reference=raw['video'].float()
                if raw.get('video_encoding')=='uint8_rgb_0_255':reference/=255
                visuals.append(dict(metadata=row,selection_reason=reasons[row['sample_id']],
                    frame_indices=indices,video=item['video'][indices],reference=reference[indices],
                    target=target[indices],prediction=maps[indices],mask=mask[indices],
                    fake_probability=probability,visual_logit=logit,map_mae=error/pixels))
    summary=dict(checkpoint_epoch=state['epoch']+1,checkpoint_file=a.checkpoint,auc_ranking='raw_logits',clips=len(results),prediction_reload_verified=True,
        map_mae=total_error/total_pixels,fake_map_mae=fake_error/fake_pixels,
        zero_prediction_fake_map_mae=zero_error/fake_pixels,
        **classification_metrics([r['label'] for r in results],[r['fake_probability'] for r in results], logits=logits))
    (report/'verified_evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    (report/'verified_predictions.json').write_text(json.dumps(results,indent=2)+'\n')
    torch.save(dict(summary=summary,examples=visuals),report/'visual_examples.pt')
    export_keys=('heads','config','epoch','checkpoint_dir','context_path','manifest','sam2_checkpoint','sam2_contract','rgb_encoder','videomae_checkpoint','videomae_contract')
    torch.save({k:state[k] for k in export_keys if k in state},report/('detector_'+a.checkpoint))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
