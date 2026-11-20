import sys,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
p=Path(sys.argv[1]);r=json.loads((p/'diagnosis.json').read_text());h=json.loads((p/'tiny_history.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(16,4.5))
v=r['methods']['combined'];names=list(v);axes[0].bar(range(len(names)),list(v.values()));axes[0].set_xticks(range(len(names)),names,rotation=40,ha='right');axes[0].set_title('Held-out fake clips: map MAE');axes[0].set_ylabel('Valid-pixel MAE (lower is better)')
f=list(r['shuffle']);x=list(range(len(f)));axes[1].bar([i-.18 for i in x],[r['shuffle'][k]['own_middle_frame_mae'] for k in f],width=.36,label='Own prediction');axes[1].bar([i+.18 for i in x],[r['shuffle'][k]['shuffled_middle_frame_mae_mean'] for k in f],width=.36,label='Shuffled within method');axes[1].set_xticks(x,f,rotation=40,ha='right');axes[1].legend();axes[1].set_title('Middle-frame correspondence control')
axes[2].plot([x['epoch'] for x in h],[x['fake_clip_mae'] for x in h],label='20 training fake clips');axes[2].set_title('Tiny-set fitting: existing architecture');axes[2].set_xlabel('Diagnostic epoch');axes[2].set_ylabel('Mean clip MAE');axes[2].legend();fig.tight_layout();fig.savefig(p/'diagnostic_summary.png',dpi=170);plt.close(fig)
vs=torch.load(p/'tiny_visuals.pt',weights_only=True,map_location='cpu');before={v['sample_id']:v for v in vs if v['epoch']==0};after=[v for v in vs if v['epoch']==100];selected=[]
for family in f:
 selected.append(next(v for v in after if v['method']==family and v['target'].max()>0))
fig,axes=plt.subplots(len(selected),4,figsize=(12,3*len(selected)))
for row,(v,ax) in enumerate(zip(selected,axes)):
 video=v['video'].float();video=video/255 if v['video'].dtype==torch.uint8 else video
 values=[video.permute(1,2,0),v['target'][0],before[v['sample_id']]['prediction'][0],v['prediction'][0]]
 for j,value in enumerate(values):
  ax[j].imshow(value.numpy(),**({} if j==0 else dict(cmap='inferno',vmin=0,vmax=1)));ax[j].set_xticks([]);ax[j].set_yticks([])
  if row==0:ax[j].set_title(['Training input','Target DSSIM','Before fitting','After 100 epochs'][j])
 ax[0].set_ylabel(v['method'])
fig.suptitle('Tiny training-set diagnostic: same architecture, fixed DSSIM scale 0–1');fig.tight_layout();fig.savefig(p/'tiny_fit_examples.png',dpi=150);plt.close(fig)
