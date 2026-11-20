import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);c=json.loads((p/'dropout_comparison.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(14,8))
fields=[('train_loss','Training loss ↓'),('classification_bce','Validation classification BCE ↓'),('accuracy','Validation accuracy ↑'),('auroc','Validation ROC-AUC ↑'),('brier','Validation Brier score ↓'),('confident_errors_99','Errors with ≥99% confidence ↓')]
for ax,(key,title) in zip(axes.flat,fields):
 for name,label,color in [('baseline_dropout_0','Dropout 0.0','#d16736'),('dropout_02','Dropout 0.2','#216b91')]:
  rows=c[name]['all_epochs'];ax.plot([r['epoch']+1 for r in rows],[r[key] for r in rows],color=color,marker='o',markersize=3,label=label)
 ax.axvline(2,color='gray',linestyle=':',linewidth=1);ax.set_title(title);ax.set_xlabel('Epoch');ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.2)
 if key in ['accuracy','auroc']:ax.set_ylim(.8,1)
 else:ax.set_ylim(bottom=0)
 ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False)
fig.suptitle('Wan-only FF++: dropout 0.0 versus 0.2',fontsize=17)
fig.text(.5,.012,'Same 250 validation entries across 50 recording groups; threshold 0.5. Dotted line: selected epoch 2 for both runs.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.085,1,.95));fig.savefig(p/'dropout_comparison.png',dpi=180);fig.savefig(p/'dropout_comparison.pdf');plt.close(fig)
