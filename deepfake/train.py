"""Train setting A using prepared face-clip tensors and frozen Wan weights."""
import argparse
import json
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from .config import DetectorConfig
from .backbone import FrozenWanFeatures
from .model import WanOnlyDetector
from .data import PreparedClipDataset, collate_clips, audit_splits
from .losses import detector_loss
from .checkpointing import commit_epoch, publish_epoch, atomic_json


def trial_forward(model, batch, feature_cache=None):
    """Cache frozen features in CPU RAM for fixed, unaugmented batch-one clips."""
    if feature_cache is None:
        return model(batch['video'], batch['frame_validity'])
    if len(batch['video']) != 1:
        raise ValueError('Feature caching currently requires batch-size 1')
    row = batch['metadata'][0]
    key = (row['split'],row['sample_id'])
    if key not in feature_cache:
        with torch.no_grad():
            grids, valid = model.extractor(batch['video'],batch['frame_validity'])
        if not feature_cache:
            print(dict(feature_cache_shapes=[list(g.shape) for g in grids],
                       feature_dtypes=[str(g.dtype) for g in grids],
                       feature_bytes_per_clip=sum(g.numel()*g.element_size() for g in grids)),flush=True)
        feature_cache[key] = ([g.detach().cpu() for g in grids],valid.cpu())
    grids, valid = feature_cache[key]
    device=batch['video'].device
    return model.heads([g.to(device) for g in grids],valid.to(device),
                       (batch['video'].shape[1],*batch['video'].shape[-2:]),batch['frame_validity'])


def classification_metrics(labels, scores, logits=None):
    y=torch.tensor(labels,dtype=torch.bool)
    s=torch.tensor(scores,dtype=torch.float64)
    if y.ndim != 1 or s.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty matching labels and scores")
    if not torch.isfinite(s).all() or (s < 0).any() or (s > 1).any():
        raise ValueError("Scores must be finite probabilities")
    rank = s if logits is None else torch.tensor(logits, dtype=torch.float64)
    if rank.shape != s.shape or not torch.isfinite(rank).all():
        raise ValueError("Invalid ranking logits")
    positive,negative=rank[y],rank[~y]
    auc=None
    if len(positive) and len(negative):
        difference=positive[:,None]-negative[None,:]
        auc=float(((difference>0).double()+.5*(difference==0).double()).mean())
    pred=s>=.5
    return dict(accuracy=float((pred==y).double().mean()),auroc=auc,
                true_positive=int((pred&y).sum()),false_negative=int((~pred&y).sum()),
                true_negative=int((~pred&~y).sum()),false_positive=int((pred&~y).sum()),
                brier=float(((s-y.double())**2).mean()),
                confident_errors_99=int(((pred!=y)&(torch.maximum(s,1-s)>=.99)).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/wan_only.json')
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--context', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume')
    p.add_argument('--selection-metric', choices=['loss', 'accuracy', 'auroc'], default='loss')
    p.add_argument('--preprocessing-config',help='Require this preprocessing fingerprint for both splits')
    p.add_argument('--cache-features',action='store_true',help='Cache frozen features in CPU RAM; batch size 1 and fixed clips only')
    a = p.parse_args()
    if a.accumulation < 1 or a.epochs < 1 or a.batch_size < 1:
        p.error('epochs, accumulation and batch-size must be positive')
    if a.cache_features and a.batch_size != 1:
        p.error('--cache-features requires --batch-size 1')
    if not torch.cuda.is_available():
        p.error('CUDA GPU is required for pretrained Wan training')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    audit_splits(a.manifest)
    cfg = DetectorConfig.load(a.config)
    expected_id = None
    if a.preprocessing_config:
        from .preprocessing_config import PreprocessingConfig
        expected_id = PreprocessingConfig.load(a.preprocessing_config).fingerprint
    rng = torch.Generator().manual_seed(a.seed)
    train = DataLoader(PreparedClipDataset(a.manifest, 'train', expected_id), batch_size=a.batch_size,
                       shuffle=True, generator=rng, collate_fn=collate_clips)
    val = DataLoader(PreparedClipDataset(a.manifest, 'val', expected_id), batch_size=a.batch_size,
                     collate_fn=collate_clips)
    extractor = FrozenWanFeatures.from_pretrained(a.checkpoint_dir, a.context, cfg)
    model = WanOnlyDetector(extractor, cfg)
    model.heads.cuda()
    opt = torch.optim.AdamW(model.heads.parameters(), lr=a.lr, weight_decay=0.01)
    start, best = 0, float('inf')
    best_accuracy, best_auroc = -1., -1.
    best_selected = float('inf') if a.selection_metric == 'loss' else -1.
    if a.resume:
        state = torch.load(a.resume, weights_only=True, map_location='cpu')
        if DetectorConfig(**state['config']).to_dict() != cfg.to_dict():
            raise ValueError('Resume configuration does not match')
        if state['checkpoint_dir'] != str(Path(a.checkpoint_dir).resolve()):
            raise ValueError('Resume backbone path does not match')
        if state['context_path'] != str(Path(a.context).resolve()):
            raise ValueError('Resume text context path does not match')
        if state['manifest'] != str(Path(a.manifest).resolve()):
            raise ValueError('Resume data manifest path does not match')
        if state.get('selection_metric', 'loss') != a.selection_metric:
            raise ValueError('Resume selection metric does not match')
        best_accuracy = state.get('best_accuracy', -1.)
        best_auroc = state.get('best_auroc', -1.)
        best_selected = state.get('best_selected', state['best'])
        model.heads.load_state_dict(state['heads'])
        opt.load_state_dict(state['optimizer'])
        start, best = state['epoch']+1, state['best']
        rng.set_state(state['loader_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
    out = Path(a.output)
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise ValueError('Nonempty run directory; choose a new output or explicitly resume')
    out.mkdir(parents=True, exist_ok=True)
    if a.resume:
        publish_epoch(out, state)
    feature_cache={} if a.cache_features else None
    run_config = dict(vars(a), detector=cfg.to_dict(), gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__))
    atomic_json(run_config, out/('resume_config.json' if a.resume else 'run_config.json'))
    for epoch in range(start, a.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        train_cls, train_count, train_error, train_pixels = 0., 0, 0., 0
        for step, batch in enumerate(train):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = trial_forward(model,batch,feature_cache)
                losses = detector_loss(outputs, batch['visual_label'], batch['map_target'],
                    batch['map_validity'], batch['frame_validity'], cfg.map_loss_weight)
            group_start = (step//a.accumulation)*a.accumulation
            count = len(batch['visual_label'])
            valid_pixels = int(losses['valid_map_pixels'])
            train_cls += float(losses['classification'].detach()) * count
            train_count += count
            train_error += float(losses['map'].detach()) * valid_pixels
            train_pixels += valid_pixels
            group_size = min(a.accumulation, len(train)-group_start)
            (losses['loss']/group_size).backward()
            if (step+1) % a.accumulation == 0 or step+1 == len(train):
                torch.nn.utils.clip_grad_norm_(model.heads.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print({'epoch': epoch, 'step': step,
                       **{k: float(v.detach()) for k,v in losses.items()}}, flush=True)
        model.eval()
        cls_sum, n, error_sum, pixels = 0., 0, 0., 0
        labels,scores,predictions,logits=[],[],[],[]
        fake_error,fake_pixels,zero_fake_error=0.,0,0.
        with torch.no_grad():
            for batch in val:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = trial_forward(model,batch,feature_cache)
                    loss = detector_loss(result, batch['visual_label'], batch['map_target'],
                        batch['map_validity'], batch['frame_validity'], cfg.map_loss_weight)
                count = len(batch['visual_label'])
                cls_sum += float(loss['classification'])*count
                n += count
                valid_pixels = int(loss['valid_map_pixels'])
                error_sum += float(loss['map'])*valid_pixels
                pixels += valid_pixels
                probability=result['visual_logit'].float().sigmoid()
                labels.extend(batch['visual_label'].cpu().tolist())
                scores.extend(probability.cpu().tolist())
                logits.extend(result['visual_logit'].float().cpu().tolist())
                mask=batch['map_validity']&batch['frame_validity'][:,:,None,None,None]
                for i,row in enumerate(batch['metadata']):
                    sample_error=(result['modification_map'][i].float()-batch['map_target'][i]).abs()
                    mae=float(sample_error[mask[i]].mean()) if mask[i].any() else None
                    predictions.append(dict(sample_id=row['sample_id'],label=int(batch['visual_label'][i]),
                        fake_probability=float(probability[i]),visual_logit=float(result['visual_logit'][i]),map_mae=mae))
                    if int(batch['visual_label'][i])==1:
                        fake_error+=float(sample_error[mask[i]].sum())
                        fake_pixels+=int(mask[i].sum())
                        zero_fake_error+=float(batch['map_target'][i][mask[i]].sum())
        score = cls_sum/n + cfg.map_loss_weight*error_sum/max(pixels,1)
        classification = classification_metrics(labels, scores, logits)
        selected = score if a.selection_metric == 'loss' else classification[a.selection_metric]
        if selected is None:
            raise ValueError('Selected metric is undefined for this validation split')
        improved = selected < best_selected if a.selection_metric == 'loss' else selected > best_selected
        checkpoint_updates = ['best.pt'] if improved else []
        if classification['accuracy'] > best_accuracy:
            best_accuracy = classification['accuracy']; checkpoint_updates.append('best_accuracy.pt')
        if classification['auroc'] is not None and classification['auroc'] > best_auroc:
            best_auroc = classification['auroc']; checkpoint_updates.append('best_auroc.pt')
        if improved:
            best_selected = selected
        best = min(score, best)
        metrics=dict(epoch=epoch,val_loss=score,classification_bce=cls_sum/n,
                     train_loss=train_cls/train_count+cfg.map_loss_weight*train_error/max(train_pixels,1),
                     train_classification_bce=train_cls/train_count,
                     train_map_mae=train_error/max(train_pixels,1),train_clips=train_count,val_clips=n,
                     map_mae=error_sum/max(pixels,1),fake_map_mae=fake_error/max(fake_pixels,1),
                     zero_prediction_fake_map_mae=zero_fake_error/max(fake_pixels,1),
                     peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                     **classification)
        state = dict(heads=model.heads.state_dict(), optimizer=opt.state_dict(),
                     epoch=epoch, best=best, config=cfg.to_dict(),
                     selection_metric=a.selection_metric, best_selected=best_selected,
                     best_accuracy=best_accuracy, best_auroc=best_auroc,
                     checkpoint_updates=checkpoint_updates, epoch_metrics=metrics,
                     epoch_predictions=predictions, checkpoint_schema=2,
                     checkpoint_dir=str(Path(a.checkpoint_dir).resolve()),
                     context_path=str(Path(a.context).resolve()), manifest=str(Path(a.manifest).resolve()),
                     loader_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate())
        commit_epoch(out, state)
        print(metrics, flush=True)


if __name__ == '__main__':
    main()
