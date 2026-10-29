"""Regenerate the principal plots from the published logs; requires matplotlib."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'learning_curves.png')
    args=ap.parse_args()
    run=ROOT/'results/lafan1_seed0'
    train=[json.loads(x) for x in (run/'train.jsonl').read_text().splitlines() if x.strip()]
    evaluation=[json.loads(x) for x in (run/'eval_bf16.jsonl').read_text().splitlines() if x.strip()]
    fp32=json.loads((run/'eval_fp32.json').read_text())['summary']
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    axes[0].plot([x['hours'] for x in evaluation],[x['norm_return'] for x in evaluation],marker='o',label='In-run bf16')
    axes[0].scatter([evaluation[-1]['hours']],[fp32['mean_ret_paper']],marker='*',s=120,label='Independent fp32')
    axes[0].set(title='Full 40-clip return',ylim=(0,1));axes[0].legend()
    axes[1].plot([x['hours'] for x in evaluation],[x['failed'] for x in evaluation],marker='o');axes[1].set_title('Failed clips / 40')
    for key,label in [('wm_err1','1-step'),('wm_errk','8-step')]:axes[2].semilogy([x['hours'] for x in train],[x[key] for x in train],label=label)
    axes[2].set_title('Normalized world-model errors');axes[2].legend()
    for ax in axes:ax.set_xlabel('Accumulated training hours');ax.grid(alpha=.25)
    fig.savefig(args.output,dpi=160);print(args.output)

if __name__=='__main__':main()
