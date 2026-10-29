"""Validate the published experiment records, without torch or a GPU."""
from pathlib import Path
import argparse,json,math,hashlib

ROOT=Path(__file__).resolve().parents[1]

def require(ok,message):
    if not ok: raise ValueError(message)

def validate_evaluation(payload):
    rows=payload['clips'];summary=payload['summary']
    require(len(rows)==summary['clips']==40,'Expected exactly 40 clips')
    require(len({r['name'] for r in rows})==40,'Duplicate clip name')
    for row in rows:
        require(row['frames']>1,'Invalid sequence length')
        require(row['steps']==row['frames']-1,f"Incomplete clip: {row['name']}")
        require(not row['failed'],f"Failed clip: {row['name']}")
        require(math.isfinite(row['ret_paper']) and math.isfinite(row['ret_sum']),'Nonfinite return')
        require(abs(row['ret_sum']-row['ret_paper']*.1*(row['frames']-1))<1e-5,'Return normalization mismatch')
    require(sum(r['frames'] for r in rows)==441120,'Unexpected data frame count')
    mean=sum(r['ret_paper'] for r in rows)/len(rows)
    weighted=sum(r['ret_sum'] for r in rows)/sum(.1*(r['frames']-1) for r in rows)
    require(abs(mean-summary['mean_ret_paper'])<1e-9,'Mean return mismatch')
    require(abs(weighted-summary['frame_weighted_ret_paper'])<1e-9,'Frame-weighted return mismatch')
    require(summary['failed']==0 and summary['success_rate']==1,'Summary failure count mismatch')
    return {'clips':40,'frames':441120,'mean_return':mean,'frame_weighted_return':weighted}

def verify(root=ROOT,checkpoint=None):
    run=root/'results/lafan1_seed0'
    result=validate_evaluation(json.loads((run/'eval_fp32.json').read_text(encoding='utf-8')))
    logs=[json.loads(line) for line in (run/'train.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require(bool(logs),'Missing training logs')
    for row in logs:
        require(all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float))),'Nonfinite training log')
    ck=json.loads((run/'checkpoint_metadata.json').read_text(encoding='utf-8'))
    summary=json.loads((root/'results/summary.json').read_text(encoding='utf-8'))
    require(ck['it']==summary['training_iterations']==30665,'Iteration mismatch')
    require(ck['updates']==summary['policy_updates']==12266,'Policy update mismatch')
    require(ck['env_steps']==summary['environment_steps']==125865984,'Environment-step mismatch')
    require(ck['cfg']['seed']==0 and ck['cfg']['resume']=='','Expected scratch seed0 run')
    require(ck['cfg']['num_envs']==4096 and ck['cfg']['horizon']==32 and ck['cfg']['k']==8,'Configuration mismatch')
    require(logs[-1]['it']<=ck['it'] and ck['it']-logs[-1]['it']<100,'Last-log/checkpoint mismatch')
    require(abs(ck['train_time']-summary['accumulated_training_seconds'])<1e-6,'Training time mismatch')
    if checkpoint:
        with Path(checkpoint).open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        require(digest==summary['checkpoint_sha256'],'Final checkpoint hash mismatch')
        result['checkpoint_hash_verified']=True
    result.update(training_log_rows=len(logs),iterations=ck['it'],scope='Single-seed in-library evaluation')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,help='Optional local final checkpoint to hash; no weights loaded')
    args=ap.parse_args()
    print(json.dumps(verify(checkpoint=args.checkpoint),indent=2))
