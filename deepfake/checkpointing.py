"""Atomic checkpoints and recoverable epoch reporting, without epoch archives."""
import json
import os
from pathlib import Path
import tempfile
import torch


def atomic_write(path, writer, binary=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb' if binary else 'w') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_save(value, path):
    atomic_write(path, lambda handle: torch.save(value, handle))


def atomic_json(value, path):
    atomic_write(path, lambda handle: json.dump(value, handle, indent=2), binary=False)


def publish_epoch(out, state):
    """Replay idempotently after last.pt commits, including interrupted reports."""
    out = Path(out)
    if 'epoch_metrics' not in state:  # Legacy checkpoint; no reconstructable report.
        return
    for name in state.get('checkpoint_updates', []):
        if name not in {'best.pt', 'best_accuracy.pt', 'best_auroc.pt'}:
            raise ValueError('Invalid checkpoint publication name')
        atomic_save(state, out/name)
    epoch = state['epoch']
    atomic_json(state['epoch_predictions'], out/f'val_predictions_epoch_{epoch:03d}.json')
    path = out/'metrics.jsonl'
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if any(row['epoch'] > epoch for row in rows):
        # Explicit rewind: preserve the old report rather than silently deleting it.
        fd, backup = tempfile.mkstemp(prefix='metrics_before_rewind_', suffix='.jsonl', dir=out)
        with os.fdopen(fd, 'w') as handle:
            handle.write(path.read_text())
    rows = [row for row in rows if row['epoch'] < epoch] + [state['epoch_metrics']]
    if len({row['epoch'] for row in rows}) != len(rows):
        raise ValueError('Duplicate epochs in metrics history')
    atomic_write(path, lambda handle: handle.write(''.join(json.dumps(r)+'\n' for r in rows)), binary=False)


def commit_epoch(out, state):
    # last.pt is the commit point. It contains enough information to repair all
    # reports and selected checkpoints if the process dies after this replace.
    atomic_save(state, Path(out)/'last.pt')
    publish_epoch(out, state)
