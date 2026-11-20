"""Full-epoch loss curves and explicitly selected held-out map visualizations."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch


def draw_examples(examples, path, temporal=False):
    panels=[]
    for example in examples:
        frames=range(len(example['frame_indices'])) if temporal else [len(example['frame_indices'])//2]
        panels.extend((example,k) for k in frames)
    fig,axes=plt.subplots(len(panels),4,figsize=(10.8,2.65*len(panels)),squeeze=False)
    scale=Normalize(0,1)
    map_colors=plt.cm.inferno.copy()
    map_colors.set_bad('#d8d8d8')
    for axes_row,(e,k) in zip(axes,panels):
        data=[e['reference'][k].permute(1,2,0),e['video'][k].permute(1,2,0),
              e['target'][k,0],e['prediction'][k,0]]
        for j,(ax,x) in enumerate(zip(axes_row,data)):
            if j<2:ax.imshow(x.numpy().clip(0,1))
            else:
                visible=np.ma.array(x.numpy(),mask=~e['mask'][k,0].numpy())
                ax.imshow(visible,cmap=map_colors,norm=scale)
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
        row=e['metadata'];label='Fake' if row['visual_label'] else 'Real'
        frame=e['frame_indices'][k]
        axes_row[0].set_ylabel(f"{label}: {row['source_video_id']}\nsegment {row['segment']['segment_index']}, frame {frame}",fontsize=9)
        probability=e['fake_probability']
        probability_text='< 0.001' if probability<.001 else '> 0.999' if probability>.999 else f'= {probability:.3f}'
        mae_text=f"{e['map_mae']:.2e}" if e['map_mae']<1e-4 else f"{e['map_mae']:.4f}"
        axes_row[1].set_xlabel(f"P(fake) {probability_text}",fontsize=9)
        axes_row[3].set_xlabel(f"Clip map MAE = {mae_text}",fontsize=9)
    for ax,title in zip(axes[0],['Paired reference','Detector input','Target DSSIM','Predicted DSSIM']):ax.set_title(title,fontsize=12,pad=10)
    fig.suptitle('Wan-only detector: held-out recording groups',fontsize=15,y=1-.10/fig.get_figheight())
    fig.subplots_adjust(left=.10,right=.90,top=1-.85/fig.get_figheight(),bottom=.90/fig.get_figheight(),wspace=.03,hspace=.22)
    cax=fig.add_axes([.918,.2,.014,.6])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=scale,cmap='inferno'),cax=cax)
    cb.ax.tick_params(labelsize=9)
    cb.set_label('DSSIM (fixed scale 0–1)',fontsize=10,labelpad=8)
    fig.text(.52,.10/fig.get_figheight(),'Reference and target are shown for evaluation only; the detector receives only the input clip.\n'
             'Map-error quantiles shown; classification errors included when present. No per-image heatmap normalization.\n'
             'Gray map borders are excluded from the loss and error metrics.',ha='center',fontsize=9)
    fig.savefig(path.with_suffix('.png'),dpi=160)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();out=a.run
    metrics=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
    summary=json.loads((out/'verified_evaluation.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    x=[r['epoch']+1 for r in metrics]
    for ax,title,tkey,vkey in zip(axes,['Total loss: BCE + map L1','Classification: BCE','Map prediction: masked L1'],
            ['train_loss','train_classification_bce','train_map_mae'],['val_loss','classification_bce','map_mae']):
        ax.plot(x,[r[tkey] for r in metrics],color='#d16736',marker='o',markersize=3,label='Training: full-epoch mean')
        ax.plot(x,[r[vkey] for r in metrics],color='#216b91',marker='o',markersize=3,label='Evaluation: all 50 clips')
        ax.axvline(summary['checkpoint_epoch'],linestyle=':',color='#777777',label=f"Selected epoch {summary['checkpoint_epoch']}")
        ax.set(title=title,xlabel='Epoch',ylabel='Loss',ylim=(0,None),xlim=(1,max(x)))
        ax.set_xticks(sorted({1,max(x),*[n for n in range(5,max(x)+1,5)]}))
        ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=10)
    fig.suptitle('Wan-only Deepfakes: 3-second segments at 8 FPS',fontsize=16)
    fig.text(.5,.025,f"Training: {metrics[0]['train_clips']:,} clips | Evaluation: 50 clips from 50 independent recording groups\n"
             'Classification is averaged over clips; map L1 is averaged over valid pixels. Evaluation selects the checkpoint.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.82,bottom=.28,left=.065,right=.98,wspace=.28)
    fig.savefig(out/'loss_curves.png',dpi=180);fig.savefig(out/'loss_curves.pdf');plt.close(fig)
    payload=torch.load(out/'visual_examples.pt',weights_only=True,map_location='cpu')
    examples=sorted(payload['examples'],key=lambda e:(-e['metadata']['visual_label'],e['map_mae']))
    draw_examples(examples,out/'heldout_examples')
    predictions=json.loads((out/'verified_predictions.json').read_text())
    fake_predictions=sorted((r for r in predictions if r['label']==1),key=lambda r:(r['map_mae'],r['sample_id']))
    temporal_ids={fake_predictions[round(.5*(len(fake_predictions)-1))]['sample_id'],fake_predictions[-1]['sample_id']}
    temporal=[e for e in examples if e['metadata']['sample_id'] in temporal_ids]
    if temporal:draw_examples(temporal,out/'heldout_temporal_examples',temporal=True)
    (out/'visual_selection.json').write_text(json.dumps([dict(sample_id=e['metadata']['sample_id'],
        reason=e['selection_reason'],fake_probability=e['fake_probability'],map_mae=e['map_mae']) for e in examples],indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
