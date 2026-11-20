"""Read-only map diagnosis: train-only templates and mismatched-clip controls.

Run with the original frozen experiment source on PYTHONPATH. No fitting uses
evaluation targets, and no training checkpoint or prepared tensor is modified.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from deepfake.config import DetectorConfig
from deepfake.backbone import FrozenWanFeatures
from deepfake.model import WanOnlyDetector
from deepfake.data import PreparedClipDataset, collate_clips, audit_splits


def off_diagonal_mean(matrix):
    return float(matrix[~np.eye(len(matrix),dtype=bool)].mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    state=torch.load(a.run/'best.pt',map_location='cpu',weights_only=True)
    manifest=Path(state['manifest']);audit_splits(manifest)
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    train=sorted((r for r in rows if r['split']=='train' and r['visual_label']==1),key=lambda r:r['sample_id'])
    middle_maps=[];total=None;pixels=None;spatial_mask=None;frames=0
    for i,row in enumerate(train):
        payload=torch.load(manifest.parent/row['tensor_path'],map_location='cpu',weights_only=True,mmap=True)
        valid=payload['frame_validity'].bool();n=int(valid.sum())
        target=payload['map_target'][:n];mask=payload['map_validity'][:n].bool()
        if spatial_mask is None:
            spatial_mask=mask[0].clone();total=torch.zeros_like(target[0],dtype=torch.float64);pixels=total.clone()
        assert torch.equal(mask,spatial_mask[None].expand_as(mask)), 'Template requires consistent valid spatial support'
        total+=(target*mask).sum(0,dtype=torch.float64);pixels+=mask.sum(0);frames+=n
        middle_maps.append(target[n//2][spatial_mask].numpy().copy())
        if (i+1)%250==0:print(json.dumps(dict(template_training_clips=i+1,total=len(train))),flush=True)
    mean=(total/pixels.clamp_min(1)).float()
    matrix=np.stack(middle_maps);del middle_maps
    median=torch.zeros_like(mean);median[spatial_mask]=torch.from_numpy(np.median(matrix,axis=0,overwrite_input=True));del matrix
    torch.save(dict(mean=mean,median=median,validity=spatial_mask,training_sample_ids=[r['sample_id'] for r in train],
                    mean_sampling='all valid training frames',median_sampling='one middle valid frame per training clip'),
               a.output/'training_templates.pt')
    print('Training-only templates complete',flush=True)
    cfg=DetectorConfig(**state['config'])
    model=WanOnlyDetector(FrozenWanFeatures.from_pretrained(state['checkpoint_dir'],state['context_path'],cfg),cfg)
    model.heads.load_state_dict(state['heads']);model.heads.cuda();model.eval()
    ds=PreparedClipDataset(manifest,'val')
    recorded={r['sample_id']:r for r in json.loads((a.run/'verified_predictions.json').read_text())}
    results=[];center_predictions=[];center_targets=[];visuals=[]
    sums=dict(model=0.,zero=0.,training_mean=0.,training_median=0.);fake_pixels=0
    temporal_pred=0.;temporal_target=0.;temporal_pixels=0
    with torch.no_grad():
        for i in range(len(ds)):
            item=ds[i];row=item['metadata'];batch=collate_clips([item]);n=int(item['frame_validity'].sum())
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(batch['video'].cuda(),batch['frame_validity'].cuda())
            pred=result['modification_map'][0].float().cpu();target=item['map_target'];mask=item['map_validity']
            score=float(result['visual_logit'].float().sigmoid()[0]);mae=float((pred-target).abs()[mask].mean())
            assert abs(score-recorded[row['sample_id']]['fake_probability'])<1e-5
            assert abs(mae-recorded[row['sample_id']]['map_mae'])<1e-5
            if row['visual_label']!=1:continue
            count=int(mask.sum());fake_pixels+=count;errors={}
            for name,value in [('model',pred),('zero',torch.zeros_like(pred)),('training_mean',mean[None].expand_as(pred)),('training_median',median[None].expand_as(pred))]:
                error=float((value-target).abs()[mask].sum());sums[name]+=error;errors[name]=error/count
            center=n//2;center_predictions.append(pred[center][spatial_mask].numpy().copy());center_targets.append(target[center][spatial_mask].numpy().copy())
            if n>1:
                temporal_mask=mask[1:n]&mask[:n-1]
                temporal_pred+=float((pred[1:n]-pred[:n-1]).abs()[temporal_mask].sum())
                temporal_target+=float((target[1:n]-target[:n-1]).abs()[temporal_mask].sum())
                temporal_pixels+=int(temporal_mask.sum())
            results.append(dict(sample_id=row['sample_id'],source_video_id=row['source_video_id'],valid_frames=n,errors=errors))
            visuals.append(dict(sample_id=row['sample_id'],source_video_id=row['source_video_id'],
                                video=item['video'][center].clone(),target=target[center].clone(),
                                prediction=pred[center].clone(),errors=errors))
    cp,ct=np.stack(center_predictions),np.stack(center_targets)
    # Rows index targets, columns index predictions. All middle frames have identical mask support.
    pair_mae=np.stack([np.abs(cp-target).mean(1) for target in ct])
    own=float(np.diag(pair_mae).mean());wrong=off_diagonal_mean(pair_mae)
    rng=np.random.default_rng(42);permuted=[];indices=np.arange(len(cp))
    for _ in range(2000):
        perm=rng.permutation(len(cp))
        while (perm==indices).any():perm=rng.permutation(len(cp))
        permuted.append(float(pair_mae[indices,perm].mean()))
    pred_correlation=np.corrcoef(cp);target_correlation=np.corrcoef(ct)
    summary=dict(checkpoint_epoch=state['epoch']+1,training_fake_clips=len(train),training_valid_frames=frames,
        evaluation_clips_verified=len(ds),evaluation_fake_clips=len(results),
        fake_only_valid_pixel_mae={key:value/fake_pixels for key,value in sums.items()},
        training_mean_sampling='all valid frames of training fake clips',
        training_median_sampling='middle valid frame of each training fake clip; equal clip weights',
        middle_frame_own_prediction_mae=own,middle_frame_wrong_clip_prediction_mae=wrong,
        own_vs_wrong_relative_mae_reduction=(wrong-own)/wrong,
        permutation_repeats=2000,permutation_p_lower_mae=(1+sum(v<=own for v in permuted))/(1+len(permuted)),
        permutation_mae_percentiles=np.percentile(permuted,[2.5,50,97.5]).tolist(),
        prediction_pairwise_spatial_pearson=off_diagonal_mean(pred_correlation),
        target_pairwise_spatial_pearson=off_diagonal_mean(target_correlation),
        adjacent_frame_prediction_l1=temporal_pred/temporal_pixels,
        adjacent_frame_target_l1=temporal_target/temporal_pixels,
        caveats=['Templates use training targets only; evaluation is not used to fit or choose a template.',
                 'Fake-only comparisons isolate map quality from real/fake classification.',
                 'Shuffling compares middle-frame maps in the shared face-crop coordinate system; consistent facial anatomy limits the control.',
                 'High similarity alone cannot prove total input independence; conditional changes in position or strength can still be useful.',
                 'DSSIM is a paired-image difference target, not a binary manipulation annotation.'])
    (a.output/'diagnosis.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'per_clip.json').write_text(json.dumps(results,indent=2)+'\n')
    np.savez_compressed(a.output/'middle_frame_controls.npz',pair_mae=pair_mae,prediction_correlation=pred_correlation,target_correlation=target_correlation,permutation_mae=np.array(permuted))
    selected=sorted(visuals,key=lambda r:(r['errors']['model'],r['sample_id']))
    torch.save(dict(mean_template=mean,median_template=median,mask=spatial_mask,
                    examples=[selected[0],selected[len(selected)//2],selected[-1]]),a.output/'diagnostic_examples.pt')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
