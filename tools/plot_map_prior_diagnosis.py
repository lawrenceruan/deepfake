"""Fixed-scale comparison of actual map predictions and a training-only prior."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    data=torch.load(a.run/'diagnostic_examples.pt',map_location='cpu',weights_only=True)
    summary=json.loads((a.run/'diagnosis.json').read_text())
    examples=data['examples'];mask=data['mask'][0].numpy();colors=plt.cm.inferno.copy();colors.set_bad('#d8d8d8')
    fig,axes=plt.subplots(len(examples),4,figsize=(12,3*len(examples)+1.1),squeeze=False)
    for row,(example,axs) in enumerate(zip(examples,axes)):
        axs[0].imshow(example['video'].permute(1,2,0).numpy().clip(0,1))
        for ax,value in zip(axs[1:],[example['target'],example['prediction'],data['median_template']]):
            ax.imshow(np.ma.array(value[0].numpy(),mask=~mask),cmap=colors,vmin=0,vmax=1)
        for ax in axs:
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
        axs[0].set_ylabel(f"Fake {example['source_video_id']}\n{['Lowest','Median','Highest'][row]} model clip error",fontsize=10)
        axs[2].set_xlabel(f"Clip MAE {example['errors']['model']:.4f}",fontsize=10)
        axs[3].set_xlabel(f"Clip MAE {example['errors']['training_median']:.4f}",fontsize=10)
    for ax,title in zip(axes[0],['Detector input','Target DSSIM','Model prediction','Fixed training median']):ax.set_title(title,fontsize=12)
    fig.suptitle('Map predictions versus a fixed training template',fontsize=16,y=.985)
    fig.subplots_adjust(left=.08,right=.90,top=.91,bottom=.16,hspace=.21,wspace=.04)
    cax=fig.add_axes([.92,.25,.014,.5]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=colors),cax=cax)
    cb.ax.tick_params(labelsize=9);cb.set_label('DSSIM, fixed scale',fontsize=10)
    errors=summary['fake_only_valid_pixel_mae']
    fig.text(.5,.035,f"All 25 validation fake clips: model MAE {errors['model']:.5f}; fixed median MAE {errors['training_median']:.5f}.\n"
             'Median template fitted only on the middle frames of 4,024 training fake clips.\n'
             'Middle validation frame shown; MAE labels use all valid frames. Gray borders are excluded.',ha='center',fontsize=10)
    fig.savefig(a.run/'map_prior_comparison.png',dpi=160);fig.savefig(a.run/'map_prior_comparison.pdf');plt.close(fig)


if __name__=='__main__':main()
