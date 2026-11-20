"""Plot recorded pilot losses without treating sampled logs as full epoch means."""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'runs/deepfakes_trial_785587'
    validation = []
    training = {}
    for job in [785575, 785587]:
        validation.extend(json.loads(line) for line in
                          (root/f'runs/deepfakes_trial_{job}/metrics.jsonl').read_text().splitlines())
        for line in (root/f'runs/deepfakes_trial_setup/train-{job}.log').read_text().splitlines():
            if not line.startswith("{'epoch':"):
                continue
            record = ast.literal_eval(line)
            if 'step' in record:
                training.setdefault(record['epoch'], []).append(record)
    validation.sort(key=lambda r:r['epoch'])
    assert [r['epoch'] for r in validation] == list(range(20))
    assert sorted(training) == list(range(20))
    for rows in training.values():
        assert sorted(r['step'] for r in rows) == list(range(0,200,10))
    summary = []
    for row in validation:
        samples = training[row['epoch']]
        summary.append(dict(epoch=row['epoch']+1, logged_training_batches=len(samples),
            training_sample_mean={k:float(np.mean([s[k] for s in samples]))
                                  for k in ['loss','classification','map']}, validation=row))
    (out/'loss_curves.json').write_text(json.dumps(dict(
        note='Training means cover 20 logged minibatches per epoch, not all 200. Validation covers all 40 held-out clips.',
        epochs=summary),indent=2)+'\n')

    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,3,figsize=(13.8,4.8))
    x = np.arange(1,21)
    for ax,title,tkey,vkey in zip(axes,
            ['Total loss: BCE + map L1','Classification loss: BCE','Map loss: masked L1'],
            ['loss','classification','map'],['val_loss','classification_bce','map_mae']):
        ax.plot(x,[r['training_sample_mean'][tkey] for r in summary],
                color='#b45a2a',linestyle='--',marker='o',markersize=3,
                label='Training (sampled log mean)')
        ax.plot(x,[r['validation'][vkey] for r in summary],
                color='#146a92',linewidth=2,marker='o',markersize=3,label='Validation (full split)')
        ax.axvline(19,color='#6b7280',linestyle=':',linewidth=1,label='Selected checkpoint: epoch 19')
        ax.set(title=title,xlabel='Epoch',ylabel='Loss',xlim=(1,20))
        ax.set_ylim(bottom=0)
        ax.set_xticks([1,5,10,15,20])
        ax.grid(alpha=.2)
    axes[2].axhline(validation[0]['zero_prediction_fake_map_mae']/2,
                   color='#5a7650',linestyle='-.',linewidth=1,
                   label='Zero-map validation baseline')
    # Equal real/fake counts and equal valid pixel counts in this fixed-size trial.
    handles,labels=axes[0].get_legend_handles_labels()
    h,l=axes[2].get_legend_handles_labels()
    fig.legend(handles+[h[-1]],labels+[l[-1]],loc='lower center',
               bbox_to_anchor=(.5,.095),ncol=2,frameon=False,fontsize=10)
    fig.suptitle('Wan-only Deepfakes pilot: recorded loss across 20 epochs',fontsize=16,y=.98)
    fig.text(.5,.035,'Training: 200 clips (100 real + 100 fake) | Validation: 40 clips (20 real + 20 fake)\n'
             'Training curves average 20 logged minibatches per epoch (every 10th batch); they are not full-epoch means.',
             ha='center',fontsize=10,color='#444444')
    fig.subplots_adjust(left=.065,right=.985,top=.84,bottom=.32,wspace=.28)
    fig.savefig(out/'loss_curves.png',dpi=180)
    fig.savefig(out/'loss_curves.pdf')
    print(json.dumps({'first_epoch':summary[0], 'last_epoch':summary[-1]},indent=2))


if __name__=='__main__':
    main()
